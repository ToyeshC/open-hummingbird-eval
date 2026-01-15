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
