import os
import json
import random
import argparse
import numpy as np
import math
import torch
import torch.nn.functional as F
from transformers import CLIPModel, AutoModel

from hbird.hbird_eval import hbird_evaluation


def seed_everything(seed: int):
    """Ensure reproducibility across torch, numpy, and python."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resize(tensor, new_g: int, old_g: int):
    """
    Resize a 2D grid tensor using:
      - bicubic interpolation when increasing size
      - area interpolation when reducing size
    """
    if new_g > old_g:
        return F.interpolate(tensor, (new_g, new_g),
                             mode="bicubic", align_corners=False)
    return F.interpolate(tensor, (new_g, new_g), mode="area")

def interpolate_pos_embed(model, img_size: int, patch_size: int) -> None:    
    """
    Resize absolute position embeddings in the model to match the new grid size (img_size // patch_size).
    Supports both standard ViT-style (with or without CLS token - the global summary token) and HuggingFace CLIP-style embeddings.
    No changes if the current grid already matches the target size or is not square.
    ToDo: what happens if not square?
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
            cls_w   = w[:1] if has_cls else None
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
                emb.position_ids = torch.arange(new_pe.shape[0],
                                                device=new_pe.device).unsqueeze(0)
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

    if any(tag in repo for tag in ("siglip2", "radio", "webssl")):
        print(f"Loading model from Hugging Face: {args.model_repo}")
        model = AutoModel.from_pretrained(
            args.model_repo,
            ignore_mismatched_sizes=True,  # allow HF to skip hard mismatches
            trust_remote_code=True,
            revision=rev,
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
            from tips.pytorch import image_encoder  # don't forget to add Tips to the the path before running this script

            print(f"Loading the TIPS model from a local repo")

            # Load the weights from one of the downloaded checkpoints
            key = repo.split("tips-")[-1]
            ckpt_dir = "../tips/pytorch/checkpoints"
            ckpt_map = {
                "s14": ("tips_oss_s14_highres_distilled_vision.npz", image_encoder.vit_small),
                "b14": ("tips_oss_b14_highres_distilled_vision.npz", image_encoder.vit_base),
                "l14": ("tips_oss_l14_highres_distilled_vision.npz", image_encoder.vit_large),
                "g14": ("tips_oss_g14_highres_vision.npz",          image_encoder.vit_giant2),
                "so400m14": ("tips_oss_so400m14_highres_largetext_distilled_vision.npz",
                            image_encoder.vit_so400m),
            }
            if key not in ckpt_map:
                raise ValueError(f"Unknown TIPS variant '{key}'")
            ckpt_path, builder = ckpt_map[key]
            weights_np = np.load(f"{ckpt_dir}/{ckpt_path}", allow_pickle=False)
            weights = {k: torch.tensor(v) for k, v in weights_np.items()}
            
            # Derive native pixel size from pos_embed
            # print("weights_np['pos_embed'].shape", weights_np["pos_embed"].shape)  # (1, 1+G^2, D)
            pos_len   = weights_np["pos_embed"].shape[1]  # 1 + G^2
            train_g   = int((pos_len - 1) ** 0.5)  # e.g. 32
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

    # --- Fallback torch.hub ---
    print(f"Loading model via torch.hub: {args.model_repo}, {name}")
    model = torch.hub.load(args.model_repo, name, pretrained=True)
    interpolate_pos_embed(model, args.input_size, args.patch_size)
    return model.eval()


def token_features(model, imgs):
    """
    Extracts patch-level features [B, N, D] from the given vision model,
    excluding CLS tokens unless required.

    Token handling logic per model type:
    - CLIP, WebSSL: return [CLS] + patch tokens → we exclude CLS
    - SigLIP, DINOv2: return only patch tokens → we use all tokens
    - RADIO: returns (summary, spatial) → we use spatial tokens
    - TIPS: returns (CLS, logits, spatial) → we use spatial tokens [B, N, D]
    """
    
    # Unwrap model if it's wrapped in DataParallel
    model = model.module if hasattr(model, "module") else model
    
    if "dinov2" in args.model_repo.lower():
        # DINOv2 returns patch tokens only (no CLS) under 'x_norm_patchtokens'
        # Shape: [B, N, D]
        return model.forward_features(imgs)['x_norm_patchtokens'], None

    elif "clip" in args.model_repo.lower():
        # CLIP returns [CLS] + patch tokens → we remove CLS
        # Shape of last_hidden: [B, N+1, D], return [B, N, D]
        vision_outputs = model.vision_model(pixel_values=imgs, output_hidden_states=True)
        last_hidden = vision_outputs.hidden_states[-1]
        return last_hidden[:, 1:], None

    elif "siglip" in args.model_repo.lower():
        # SigLIP returns only patch tokens (no CLS)
        # Shape: [B, N, D]
        vision_outputs = model.vision_model(pixel_values=imgs, output_hidden_states=True)
        last_hidden = vision_outputs.hidden_states[-1]
        return last_hidden, None

    elif "radio" in args.model_repo.lower():
        # RADIO returns (summary, spatial) → use spatial tokens only
        # Shape: [B, N, D]
        _, spatial_features = model(imgs)
        return spatial_features, None

    elif "webssl" in args.model_repo.lower():
        # WebSSL returns [CLS] + patch tokens → remove CLS
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
        # Default fallback: assumes ViT-style [CLS] + patch tokens → remove CLS
        # Shape: [B, N+1, D] → return [B, N, D]
        return model.get_intermediate_layers(imgs)[0][:, 1:], None

def main(args):
    print(f"The script arguments are {args}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args)
    
    if torch.cuda.device_count() > 1:  # make all GPUs work in parallel on the batch (if more than 1 GPU is available)
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs.")
        model = torch.nn.DataParallel(model)

    model = model.to(device)

    if hasattr(model, "interpolate_pos_encoding"):
        print("The model is loaded from Hugging Face and supports auto embedding interpolation, \n"
              "however manual embedding interpolation will be applied if the input_size provided does not match \n" \
              "the input_size on which the model was trained.")
    else:
        print("The model does not support automatic embedding interpolation, \n"
              "but manual embedding interpolation will be applied if the input_size provided does not match \n" \
              "the input_size on which the model was trained.")
        
    if args.nn_params:
        try:
            nn_params = json.loads(args.nn_params)
        except json.JSONDecodeError:
            raise ValueError("Invalid format for --nn_params. Provide a valid JSON string.")
    else:
        nn_params = {}
    
    # Decide whether to enable FAISS sharding (moves faiss index to multiple GPUs and helps with OOM errors)
    num_gpus = torch.cuda.device_count()
    if str(device) == "cuda" and args.nn_method == "faiss" and num_gpus > 1:
        print(f"Detected {num_gpus} GPUs. Enabling FAISS index sharding.")
        nn_params.setdefault("idx_shard", True)
    else:
        print(f"FAISS sharding not used (device: {device}, GPUs available: {num_gpus})")


    # Handle MVImgNet angle bin parsing
    if args.dataset_name.lower() == "mvimgnet":
        assert args.train_bins is not None, "You must specify --train_bins for mvimgnet."
        assert args.val_bins is not None, "You must specify --val_bins for mvimgnet."

        train_bins_list = args.train_bins.split(',')
        val_bins_list = args.val_bins.split(',')

        assert type(train_bins_list) == list, "train_bins must be a comma-separated list."
        assert type(val_bins_list) == list, "val_bins must be a comma-separated list."

        print(f"📦 MVImgNet → Train bins: {train_bins_list}")
        print(f"📦 MVImgNet → Val bins:   {val_bins_list}")

        hbird_miou_list = hbird_evaluation(
            model=model,
            d_model=args.d_model,
            patch_size=args.patch_size,
            batch_size=args.batch_size,
            input_size=args.input_size,
            augmentation_epoch=args.augmentation_epoch,
            device=device,
            return_knn_details=args.return_knn_details,
            nn_method=args.nn_method,
            n_neighbours=args.n_neighbours,
            nn_params=nn_params,
            ftr_extr_fn=token_features,
            dataset_name=args.dataset_name,
            data_dir=args.data_dir,
            memory_size=args.memory_size,
            num_workers=args.num_workers,
            train_fs_path=args.train_fs_path,
            val_fs_path=args.val_fs_path,
            train_bins=train_bins_list,
            val_bins=val_bins_list,
        )
        print(f"val_bin(s) : {val_bins_list},  Hummingbird Evaluation (mIoU): {np.mean(hbird_miou_list)}")
        print(f"Hummingbird Evaluation (mIoU) per class: {hbird_miou_list}")

    else:
        hbird_miou = hbird_evaluation(
            model=model,
            d_model=args.d_model,
            patch_size=args.patch_size,
            batch_size=args.batch_size,
            input_size=args.input_size,
            augmentation_epoch=args.augmentation_epoch,
            device=device,
            return_knn_details=args.return_knn_details,
            nn_method=args.nn_method,
            n_neighbours=args.n_neighbours,
            nn_params=nn_params,
            ftr_extr_fn=token_features,
            dataset_name=args.dataset_name,
            data_dir=args.data_dir,
            memory_size=args.memory_size,
            num_workers=args.num_workers,
            train_fs_path=args.train_fs_path,
            val_fs_path=args.val_fs_path,
        )
        print(f"Hummingbird Evaluation (mIoU): {hbird_miou}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HummingBird Evaluation")

    # Reproducibility
    parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility")

    # Model arguments
    parser.add_argument("--model_repo", default=None, type=str,
                        help="Torch Hub repo or HuggingFace repo ID")
    parser.add_argument("--model_name", default=None, type=str,
                        help="Model name for torch.hub (e.g. dino_vits16)")
    parser.add_argument("--d_model", default=None, type=int,
                        help="Size of the embedding feature vectors")
    parser.add_argument("--revision", default=None, type=str,
                    help="(HuggingFace only) Commit hash, tag, or branch to pin model version")

    # Input & patching
    parser.add_argument("--input_size", default=None, type=int, help="Size of the input image")
    parser.add_argument("--patch_size", default=None, type=int, help="Size of the model patch")

    # Dataset arguments
    parser.add_argument("--data_dir", default=None, type=str,
                        help="Path to the dataset root")
    parser.add_argument("--dataset_name", default=None, type=str, help="Dataset name (e.g. voc, mvimgnet)")
    parser.add_argument("--train_fs_path", default=None, type=str,
                        help="Path to train file list")
    parser.add_argument("--val_fs_path", default=None, type=str,
                        help="Path to validation file list")
    # MVImgNet Dataset specific arguments
    parser.add_argument("--train_bins", type=str, default=None,
                        help="(MVImgNet only) Training angle bins as comma-sep list like 0,15,30")
    parser.add_argument("--val_bins", type=str, default=None,
                        help="(MVImgNet only) Validation angle bins as comma-sep list like 0,15,30")

    # Evaluation behavior
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size for evaluation")
    parser.add_argument("--augmentation_epoch", default=1, type=int,
                        help="Number of augmentation passes over training data")
    parser.add_argument("--memory_size", default=None, type=int,
                        help="Optional memory size cap")
    parser.add_argument("--num_workers", default=None, type=int, help="Num workers for DataLoader")

    # Nearest neighbor search
    parser.add_argument("--n_neighbours", default=30, type=int,
                        help="Number of neighbors to use in k-NN search")
    parser.add_argument("--nn_method", default="faiss", type=str,
                        help="Method for nearest neighbor search")
    parser.add_argument("--nn_params", default=None, type=str,
                        help="JSON string for nearest neighbor parameters")
    parser.add_argument("--return_knn_details", action=None,
                        help="Whether to return details of k-NN results")

    args = parser.parse_args()

    seed_everything(args.seed)

    main(args)
