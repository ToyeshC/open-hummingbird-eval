import math
import torch
import torch.nn as nn

def token_features(args, model, imgs):
    """
    Extracts patch-level features [B, N, D] from the given vision model,
    excluding CLS tokens unless required.

    Supported models and output token handling:
    - CLIP, WebSSL: return [CLS] + patch tokens, we exclude CLS
    - SigLIP, DINOv2: return only patch tokens, we use all tokens
    - DINOv3: return [CLS] + register + patch tokens, we exclude CLS and register tokens
    - RADIO: returns (summary, spatial), we use spatial tokens
    - TIPS: returns (CLS, logits, spatial), we use spatial tokens [B, N, D]
    - ViT (default): return [CLS] + patch tokens, we exclude CLS
    """
    # Unwrap model if it's wrapped in DataParallel
    model = model.module if hasattr(model, "module") else model

    if "dinov2" in args.model_repo.lower():
        # DINOv2 returns patch tokens only (no CLS) under 'x_norm_patchtokens'
        # Shape: [B, N, D]
        return model.forward_features(imgs)["x_norm_patchtokens"], None

    elif "dinov3" in args.model_repo.lower():
        # HF DINOv3: last_hidden_state has shape [B, 1+R+N, D]
        # where 1 = CLS token, R = register tokens, N = patch tokens
        # Shape: [B, N, D], we keep only the patch tokens
        out = model(pixel_values=imgs, output_hidden_states=True)
        R = getattr(model.config, "num_register_tokens", 0)
        return out.last_hidden_state[:, 1 + R :, :], None

    elif "clip" in args.model_repo.lower():
        # CLIP returns [CLS] + patch tokens so we remove CLS
        # Shape of last_hidden: [B, N+1, D], return [B, N, D]
        vision_outputs = model.vision_model(
            pixel_values=imgs, output_hidden_states=True
        )
        last_hidden = vision_outputs.hidden_states[-1]
        return last_hidden[:, 1:], None

    elif "siglip" in args.model_repo.lower():
        # SigLIP returns only patch tokens (no CLS)
        # Shape: [B, N, D]
        vision_outputs = model.vision_model(pixel_values=imgs)
        last_hidden = vision_outputs.last_hidden_state
        return last_hidden, None

    elif "radio" in args.model_repo.lower():
        # RADIO returns (summary, spatial) so we use spatial tokens only
        # Shape: [B, N, D]
        _, spatial_features = model(imgs)
        return spatial_features, None

    elif "webssl" in args.model_repo.lower():
        # WebSSL returns [CLS] + patch tokens so we remove CLS
        # Shape of last_hidden_state: [B, N+1, D], return [B, N, D]
        outputs = model(pixel_values=imgs, output_hidden_states=True)
        last_hidden = outputs.last_hidden_state
        return last_hidden[:, 1:], None

    elif "tips" in args.model_repo.lower():
        # TIPS returns (cls_tokens, logits, spatial_tokens)
        # spatial_tokens shape: [B, N, D] — already flattened
        # We exclude CLS and use spatial tokens only
        output = model(imgs)
        patch_tokens = output[2]  # [B, N, D]
        return patch_tokens, None

    else:
        # Default fallback: assumes ViT-style [CLS] + patch tokens so we remove CLS
        # Shape: [B, N+1, D] so return [B, N, D]
        return model.get_intermediate_layers(imgs)[0][:, 1:], None


def load_vggt(
    backbone: str = "vggt-500m",
    ckpt_path: str | None = None,
    device: torch.device | str = "cpu",
    hf_model_id: str | None = None,
):
    """
    Load a VGGT model either from Hugging Face or a local checkpoint.
    """
    try:
        from vggt.models.vggt import VGGT  # type: ignore
    except Exception:
        try:
            from vggt.vggt import VGGT  # type: ignore
        except Exception as exc:
            raise ImportError(
                "Could not import VGGT. Ensure the vggt package is installed."
            ) from exc

    if ckpt_path is None:
        if hf_model_id is None:
            b = backbone.lower()
            if "1b" in b:
                hf_model_id = "facebook/VGGT-1B"
            elif "500m" in b or "500" in b:
                hf_model_id = "facebook/VGGT-1B"
            elif "200m" in b or "200" in b:
                hf_model_id = "facebook/VGGT-1B"
            else:
                hf_model_id = "facebook/VGGT-1B"
        model = VGGT.from_pretrained(hf_model_id)
    else:
        model = VGGT(backbone=backbone)
        state = torch.load(ckpt_path, map_location="cpu")
        state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
        model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    model.eval()
    return model


class VGGTFeatureExtractor(nn.Module):
    """
    Adapter that exposes VGGT aggregator tokens as patch embeddings
    compatible with Hummingbird's evaluation pipeline.
    """

    def __init__(
        self,
        vggt_model: nn.Module,
        eval_spatial_resolution: int,
        d_model: int | None = None,
        normalize: bool = False,
    ):
        super().__init__()
        self.model = vggt_model
        self.eval_spatial_resolution = eval_spatial_resolution
        self.d_model = d_model if d_model is not None else -1
        self.normalize = normalize

    @property
    def device(self):
        return next(self.model.parameters()).device

    @torch.no_grad()
    def forward_features(self, imgs: torch.Tensor):
        """
        Args:
            imgs: Tensor (B, 3, H, W) - single view images
        Returns:
            features: (B, P, D), None
        """
        images = imgs
        if self.normalize:
            images = images

        if images.dim() == 4:
            images = images.unsqueeze(1)

        use_cuda = images.is_cuda
        amp_dtype = (
            torch.bfloat16
            if (use_cuda and torch.cuda.get_device_capability()[0] >= 8)
            else torch.float16
        )
        with torch.cuda.amp.autocast(enabled=use_cuda, dtype=amp_dtype):
            aggregated_tokens_list, _ = self.model.aggregator(images)

        tokens = aggregated_tokens_list[-1]
        if tokens.dim() == 4:
            b, s, p, d = tokens.shape
            tokens = tokens.view(b, s * p, d)

        expected_patches = self.eval_spatial_resolution * self.eval_spatial_resolution
        if tokens.shape[1] > expected_patches:
            tokens = tokens[:, :expected_patches, :]

        if self.d_model == -1:
            self.d_model = tokens.shape[-1]

        grid = int(math.sqrt(tokens.shape[1]))
        if grid != self.eval_spatial_resolution:
            pass

        return tokens, None
