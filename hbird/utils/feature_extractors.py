import math
from typing import Dict, Optional

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
    def forward_features(
        self,
        imgs: torch.Tensor,
        camera: Optional[Dict[str, torch.Tensor]] = None,
        return_geometry: bool = True,
        query_index: int = 0,
    ):
        """
        Args:
            imgs: Tensor (B, 3, H, W) for single view or (B, S, 3, H, W) for multi-view
                  or dict with "views" key containing (B, S, 3, H, W)
        Returns:
            features: (B, P, D_with_geometry), extras
        """
        images = imgs if not isinstance(imgs, dict) else imgs["views"]
        if self.normalize:
            images = images

        # Ensure images have sequence dimension: (B, S, C, H, W)
        if images.dim() == 4:
            # Single view: (B, C, H, W) -> add sequence dimension
            images = images.unsqueeze(1)  # (B, 1, C, H, W)
        elif images.dim() == 5:
            # Multi-view: (B, S, C, H, W) - already correct
            pass
        else:
            raise ValueError(f"Unexpected input shape: {images.shape}, expected 4D (B,C,H,W) or 5D (B,S,C,H,W)")

        use_cuda = images.is_cuda
        amp_dtype = (
            torch.bfloat16
            if (use_cuda and torch.cuda.get_device_capability()[0] >= 8)
            else torch.float16
        )
        # VGGT aggregator expects (B, S, C, H, W) where S is sequence length
        # The aggregator performs cross-view attention and token fusion
        if use_cuda:
            with torch.amp.autocast(device_type='cuda', dtype=amp_dtype):
                aggregated_tokens_list, patch_start_idx = self.model.aggregator(images)
        else:
            aggregated_tokens_list, patch_start_idx = self.model.aggregator(images)

        tokens = aggregated_tokens_list[-1]  # (B, S, P_total, 2C)
        if tokens.dim() != 4:
            raise ValueError(f"Unexpected VGGT token shape {tokens.shape}")

        patch_tokens = tokens[:, :, patch_start_idx:, :]
        if query_index >= patch_tokens.shape[1]:
            raise IndexError(f"Query index {query_index} out of range for sequence {patch_tokens.shape[1]}")
        query_tokens = patch_tokens[:, query_index, :, :]

        features = query_tokens.float()
        extras: Dict[str, torch.Tensor] = {}

        predictions: Dict[str, torch.Tensor] = {}
        if return_geometry:
            # Extract geometry predictions (depth, 3D points, camera pose) from VGGT heads
            # These are computed from the aggregated multi-view tokens
            if use_cuda:
                with torch.amp.autocast(device_type='cuda', dtype=amp_dtype):
                    if getattr(self.model, "camera_head", None) is not None:
                        pose_enc_list = self.model.camera_head(aggregated_tokens_list)
                        predictions["pose_enc"] = pose_enc_list[-1]
                    if getattr(self.model, "depth_head", None) is not None:
                        depth, depth_conf = self.model.depth_head(
                            aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                        )
                        predictions["depth"] = depth
                        predictions["depth_conf"] = depth_conf
                    if getattr(self.model, "point_head", None) is not None:
                        world_pts, world_conf = self.model.point_head(
                            aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                        )
                        predictions["world_points"] = world_pts
                        predictions["world_points_conf"] = world_conf
            else:
                if getattr(self.model, "camera_head", None) is not None:
                    pose_enc_list = self.model.camera_head(aggregated_tokens_list)
                    predictions["pose_enc"] = pose_enc_list[-1]
                if getattr(self.model, "depth_head", None) is not None:
                    depth, depth_conf = self.model.depth_head(
                        aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                    )
                    predictions["depth"] = depth
                    predictions["depth_conf"] = depth_conf
                if getattr(self.model, "point_head", None) is not None:
                    world_pts, world_conf = self.model.point_head(
                        aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                    )
                    predictions["world_points"] = world_pts
                    predictions["world_points_conf"] = world_conf

        if predictions:
            extras["vggt_predictions"] = {k: v.float() for k, v in predictions.items()}
            depth = predictions.get("depth")
            world_points = predictions.get("world_points")
            # Enhance patch features with geometry information from VGGT predictions
            # This leverages VGGT's multi-view geometry understanding
            if depth is not None:
                # Extract depth for query view and pool to patch resolution
                depth_query = depth[:, query_index, ..., 0]  # (B, H, W)
                depth_query = depth_query.unsqueeze(1)  # (B,1,H,W)
                depth_tokens = torch.nn.functional.adaptive_avg_pool2d(
                    depth_query.float(), (self.eval_spatial_resolution, self.eval_spatial_resolution)
                )
                depth_tokens = depth_tokens.flatten(2).transpose(1, 2)  # (B, P, 1)
                # Normalize depth tokens for better feature fusion
                depth_tokens = (depth_tokens - depth_tokens.mean(dim=1, keepdim=True)) / (depth_tokens.std(dim=1, keepdim=True) + 1e-8)
                features = torch.cat([features, depth_tokens], dim=-1)
            if world_points is not None:
                # Extract 3D world coordinates for query view and pool to patch resolution
                world_query = world_points[:, query_index]  # (B,H,W,3)
                world_query = world_query.permute(0, 3, 1, 2).contiguous()  # (B,3,H,W)
                world_tokens = torch.nn.functional.adaptive_avg_pool2d(
                    world_query.float(), (self.eval_spatial_resolution, self.eval_spatial_resolution)
                )
                world_tokens = world_tokens.flatten(2).transpose(1, 2)  # (B,P,3)
                # Normalize world point tokens for better feature fusion
                world_tokens = (world_tokens - world_tokens.mean(dim=1, keepdim=True)) / (world_tokens.std(dim=1, keepdim=True) + 1e-8)
                features = torch.cat([features, world_tokens], dim=-1)

        camera_extras: Dict[str, torch.Tensor] = {}
        if camera is not None:
            camera_extras = {
                k: v.to(features.device, dtype=torch.float32) for k, v in camera.items() if isinstance(v, torch.Tensor)
            }
            extras["camera"] = camera_extras

            if "cam_to_world" in camera_extras and "world_points" in predictions:
                # Align predicted world points relative to camera origin
                cam_pose = camera_extras["cam_to_world"][:, query_index]  # (B,4,4)
                cam_origin = cam_pose[:, :3, 3].unsqueeze(1)  # (B,1,3)
                world_query = predictions["world_points"][:, query_index].reshape(
                    features.shape[0], -1, 3
                )  # (B,H*W,3)
                relative_world = world_query - cam_origin
                relative_world = relative_world.mean(dim=1, keepdim=True)
                extras["relative_world_mean"] = relative_world

        expected_patches = self.eval_spatial_resolution * self.eval_spatial_resolution
        if features.shape[1] != expected_patches:
            raise ValueError(
                f"VGGT produced {features.shape[1]} tokens, expected {expected_patches} "
                f"for spatial resolution {self.eval_spatial_resolution}"
            )

        if self.d_model == -1 or self.d_model != features.shape[-1]:
            self.d_model = features.shape[-1]

        return features, extras if extras else None
