import os
import numpy as np
import math
import torch
import torch.nn.functional as F
from transformers import CLIPModel, AutoModel
from hbird.utils.feature_extractors import load_vggt

def _resize(tensor, new_g: int, old_g: int):
    """
    Resize a 2D grid tensor using:
      - bicubic interpolation when increasing size
      - area interpolation when reducing size
    """
    if new_g > old_g:
        return F.interpolate(
            tensor, (new_g, new_g), mode="bicubic", align_corners=False
        )
    return F.interpolate(tensor, (new_g, new_g), mode="area")


def interpolate_pos_embed(model, img_size: int, patch_size: int) -> None:
    """
    Resize absolute position embeddings in the model to match the new grid size (img_size // patch_size).
    Supports both standard ViT-style (with or without CLS token - the global summary token) and HuggingFace CLIP-style embeddings.
    No changes if the current grid already matches the target size or is not square.

    Note: anything but Tips can be interpolated automatically as well.
    """
    new_grid = img_size // patch_size

    # --- DINO / custom ViT style ---
    pos = getattr(model, "pos_embed", None)
    if pos is not None:
        has_cls = pos.shape[1] % 2 == 1
        patch_tok = pos[:, 1:] if has_cls else pos
        seq = patch_tok.shape[1]
        old_grid = int(math.sqrt(seq))

        if old_grid * old_grid == seq and old_grid != new_grid:
            # Split CLS + patches
            cls_tok = pos[:, :1] if has_cls else None

            # Interpolate
            x = patch_tok.reshape(1, old_grid, old_grid, -1).permute(0, 3, 1, 2)
            x = _resize(x, new_grid, old_grid)
            x = x.permute(0, 2, 3, 1).reshape(1, -1, x.shape[1])

            # Rebuild
            new_pos = torch.cat([cls_tok, x], dim=1) if has_cls else x
            model.pos_embed = torch.nn.Parameter(new_pos.detach())

    # --- HuggingFace CLIP style ---
    if hasattr(model, "vision_model"):
        emb = model.vision_model.embeddings
        pe = getattr(emb, "position_embedding", None)

        if pe is not None:
            w = pe.weight  # (1+N, D)
            has_cls = w.shape[0] % 2 == 1
            cls_w = w[:1] if has_cls else None
            patch_w = w[1:] if has_cls else w
            old_grid = int(math.sqrt(patch_w.shape[0]))

            if old_grid * old_grid == patch_w.shape[0] and old_grid != new_grid:
                # Interpolate
                y = patch_w.reshape(old_grid, old_grid, -1).permute(2, 0, 1)[None]
                y = _resize(y, new_grid, old_grid)
                y = y.permute(0, 2, 3, 1).reshape(-1, y.shape[1])

                new_pe = torch.cat([cls_w, y], dim=0) if has_cls else y
                emb.position_embedding.weight = torch.nn.Parameter(new_pe.detach())

                # Sync buffers
                emb.position_ids = torch.arange(
                    new_pe.shape[0], device=new_pe.device
                ).unsqueeze(0)
                emb.position_embedding.num_embeddings = new_pe.shape[0]

            # Lift CLIP's hard guard
            emb.image_size = img_size
            model.vision_model.config.image_size = img_size
            if hasattr(model.config, "vision_config"):
                model.config.vision_config.image_size = img_size


def load_model(args):
    """
    Load a vision model from HuggingFace, torch.hub, or a local TIPS repo,
    and interpolate (resize) positional embeddings to match the input size.
    """
    repo = args.model_repo.lower()
    name = getattr(args, "model_name", "")
    rev = getattr(args, "revision", None)

    if args.input_size % args.patch_size:
        raise ValueError("input_size must be divisible by patch_size")

    # --- Loaded from HuggingFace ---
    if "clip" in repo:
        print(f"Loading model from Hugging Face: {args.model_repo}")
        model = CLIPModel.from_pretrained(
            args.model_repo,
            revision=rev,
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
        )

        # Resize any absolute pos-embeds
        interpolate_pos_embed(model, args.input_size, args.patch_size)

        # Patch CLIP's internal guard
        emb = model.vision_model.embeddings
        emb.image_size = args.input_size
        model.vision_model.config.image_size = args.input_size
        model.config.vision_config.image_size = args.input_size

        return model.eval()

    if any(tag in repo for tag in ("siglip2", "radio", "webssl", "dinov3")):
        print(f"Loading model from Hugging Face: {args.model_repo}")
        model = AutoModel.from_pretrained(
            args.model_repo,
            ignore_mismatched_sizes=True,  # allow HF to skip hard mismatches
            trust_remote_code=True,
            revision=rev,
            token=os.getenv("HUGGINGFACE_HUB_TOKEN"),
        )
        interpolate_pos_embed(model, args.input_size, args.patch_size)
        return model.eval()

    # --- Loaded from torch.hub ---
    if "dinov2" in repo or "dino" in repo:
        print(f"Loading model via torch.hub: {args.model_repo}, {name}")
        model = torch.hub.load(args.model_repo, name, pretrained=True)
        interpolate_pos_embed(model, args.input_size, args.patch_size)
        return model.eval()

    # --- Loaded from local TIPS repo ---
    elif "tips" in repo.lower():
        try:
            # Tips must be set up before running this
            from tips.pytorch import (image_encoder)

            print("Loading the TIPS model from a local repo")

            # Load the weights from one of the downloaded checkpoints
            key = repo.split("tips-")[-1]
            ckpt_dir = f"{os.getenv('HOME')}/tips/pytorch/checkpoints"
            ckpt_map = {
                "s14": (
                    "tips_oss_s14_highres_distilled_vision.npz",
                    image_encoder.vit_small,
                ),
                "b14": (
                    "tips_oss_b14_highres_distilled_vision.npz",
                    image_encoder.vit_base,
                ),
                "l14": (
                    "tips_oss_l14_highres_distilled_vision.npz",
                    image_encoder.vit_large,
                ),
                "g14": ("tips_oss_g14_highres_vision.npz", image_encoder.vit_giant2),
                "so400m14": (
                    "tips_oss_so400m14_highres_largetext_distilled_vision.npz",
                    image_encoder.vit_so400m,
                ),
            }
            if key not in ckpt_map:
                raise ValueError(f"Unknown TIPS variant '{key}'")
            ckpt_path, builder = ckpt_map[key]
            weights_np = np.load(f"{ckpt_dir}/{ckpt_path}", allow_pickle=False)
            weights = {k: torch.tensor(v) for k, v in weights_np.items()}

            # Derive native pixel size from pos_embed
            # print("weights_np['pos_embed'].shape", weights_np["pos_embed"].shape)  # (1, 1+G^2, D)
            pos_len = weights_np["pos_embed"].shape[1]  # 1 + G^2
            train_g = int((pos_len - 1) ** 0.5)  # e.g. 32
            native_px = train_g * args.patch_size  # 32x14 = 448
            # print(f"Training grid size: {train_g} (px)")
            # print(f"Training image size: {native_px} (px)")

            # Build backbone at native resolution so all shapes match
            model = builder(
                img_size=native_px,
                patch_size=args.patch_size,
                block_chunks=0,  # flat layout used in released ckpts
                ffn_layer="mlp",
                init_values=1.0,
                interpolate_antialias=True,
            )
            model.load_state_dict(weights)

            # Interpolate the embeddings to args.input_size
            interpolate_pos_embed(model, args.input_size, args.patch_size)
            return model.eval()

        except ImportError:
            raise ImportError(
                "Could not import TIPS. Ensure the TIPS repo is cloned from "
                "https://github.com/google-deepmind/tips and its path is in PYTHONPATH."
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"TIPS checkpoint not found at {ckpt_path}. "
                "Make sure you've run 'download_checkpoints.sh' in the TIPS repo."
            )

    elif "vggt" in repo.lower():
        # VGGT path
        model = load_vggt(
            backbone=args.model,
            ckpt_path=args.vggt_ckpt,
            # device=device,
            hf_model_id=args.vggt_hf_id,
        )
        return model.eval()
        # ToDo
    
        

    # --- Fallback torch.hub ---
    print(f"Loading model via torch.hub: {args.model_repo}, {name}")
    model = torch.hub.load(args.model_repo, name, pretrained=True)
    interpolate_pos_embed(model, args.input_size, args.patch_size)
    return model.eval()

