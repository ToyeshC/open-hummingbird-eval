import os
import json
import random
import argparse
import numpy as np
import math
import csv
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import CLIPModel, AutoModel

from hbird.hbird_eval import hbird_evaluation

RESULTS_PATH = 'results/results_exp_a_500_sharding_batch4_workers8_dataparallel_memory10240000.csv'
JOB_ID = os.environ.get('SLURM_JOB_ID')
VAL_BINS = [0, 15, 30, 45, 60, 75, 90]
TRAIN_BIN_LISTS = [
    [0, 30, 60, 90],
    [0, 45, 90],
    [0, 90],
    [0],
]
# Setup device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def r3(x, to=3):
    """
    Round x to 3 (default) decimals.
    """

    return round(x, to)


def seed_everything(seed: int):
    """Ensure reproducibility across torch, numpy, and python."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# --------- old, no interpolation ----------
# def load_model(args):
#     """
#     Load a vision model based on the source.

#     - CLIP, SigLIP, RADIO, WebSSL: from Hugging Face (transformers)
#     - DINOv2: from torch.hub
#     - DINO: from torch.hub
#     - TIPS: from local repo (requires cloning the TIPS repo)
#     """
#     repo = args.model_repo
#     revision = getattr(args, "revision", None)
#     model_name = getattr(args, "model_name", None)

#     # Load CLIP (ViT encoder + projection head)
#     if "clip" in repo.lower():
#         from transformers import CLIPModel
#         print(f"Loading CLIP model from HF Hub: {repo}")
#         model = CLIPModel.from_pretrained(repo, revision=revision, trust_remote_code=True)
#         model.eval()
#         return model
    
#     if "siglip2" in repo.lower():
#         from transformers import AutoModel
#         print(f"Loading SigLIP2 model from HF Hub: {repo}")
#         model = AutoModel.from_pretrained(
#             repo,
#             revision=revision,
#             trust_remote_code=True,
#         )
#         return model.eval()

#     # Load SigLIP (ViT + MAP head, no CLS token)
#     elif "siglip" in repo.lower():
#         from transformers import SiglipModel
#         print(f"Loading SigLIP model from HF Hub: {repo}")
#         model = SiglipModel.from_pretrained(repo, revision=revision, trust_remote_code=True)
#         model.eval()
#         return model

#     # Load RADIO or other custom transformer models from HF
#     elif "radio" in repo.lower():
#         from transformers import AutoModel
#         print(f"Loading RADIO model from HF Hub: {repo}")
#         model = AutoModel.from_pretrained(repo, revision=revision, trust_remote_code=True)
#         model.eval()
#         return model

#     # Load DINOv2 from torch.hub
#     elif "dinov2" in repo.lower():
#         import torch
#         print(f"Loading DINOv2 model via torch.hub: {repo}, {model_name}")
#         model = torch.hub.load(repo, model_name, pretrained=True)
#         model.eval()
#         return model

#     # Load original DINO from torch.hub
#     elif "dino" in repo.lower():
#         import torch
#         print(f"Loading original DINO model via torch.hub: {repo}, {model_name}")
#         model = torch.hub.load(repo, model_name, pretrained=True)
#         model.eval()
#         return model
    
#     # Load TIPS from local repo (using pre-downloaded .npz checkpoints)
#     elif "tips" in repo.lower():
#         try:
#             import numpy as np
#             import torch
#             from tips.pytorch import image_encoder

#             print(f"Loading TIPS model: {repo}")
#             variant_key = repo.lower().split("tips-")[-1]

#             # Map model variants to checkpoints and corresponding constructor functions
#             checkpoint_map = {
#                 "s14": ("../tips/pytorch/checkpoints/tips_oss_s14_highres_distilled_vision.npz", image_encoder.vit_small),
#                 "b14": ("../tips/pytorch/checkpoints/tips_oss_b14_highres_distilled_vision.npz", image_encoder.vit_base),
#                 "l14": ("../tips/pytorch/checkpoints/tips_oss_l14_highres_distilled_vision.npz", image_encoder.vit_large),
#                 "g14": ("../tips/pytorch/checkpoints/tips_oss_g14_highres_vision.npz", image_encoder.vit_giant2),
#                 "so400m14": ("../tips/pytorch/checkpoints/tips_oss_so400m14_highres_largetext_distilled_vision.npz", image_encoder.vit_so400m),
#             }

#             if variant_key not in checkpoint_map:
#                 raise ValueError(f"Unknown TIPS variant '{variant_key}'. Expected one of: {list(checkpoint_map.keys())}")

#             checkpoint_path, model_fn = checkpoint_map[variant_key]

#             # Load the .npz checkpoint and convert to PyTorch tensors
#             weights_np = np.load(checkpoint_path, allow_pickle=False)
#             weights = {}
#             for k, v in weights_np.items():
#                 weights[k] = torch.tensor(v)

#             # Initialize and load the vision transformer
#             model = model_fn(
#                 img_size=args.input_size,
#                 patch_size=args.patch_size,
#                 ffn_layer='mlp',
#                 block_chunks=0,
#                 init_values=1.0,
#                 interpolate_antialias=True,
#                 interpolate_offset=0.0,
#             )
#             model.load_state_dict(weights)
#             model.eval()
#             return model

#         except ImportError:
#             raise ImportError(
#                 "Could not import TIPS. Ensure the TIPS repo is cloned from "
#                 "https://github.com/google-deepmind/tips and its path is in PYTHONPATH."
#             )
#         except FileNotFoundError:
#             raise FileNotFoundError(
#                 f"TIPS checkpoint not found at {checkpoint_path}. "
#                 "Make sure you've run `download_checkpoints.sh` in the TIPS repo."
#             )

#     # Load WebSSL from Hugging Face
#     elif "webssl" in repo.lower():
#         from transformers import AutoModel
#         print(f"Loading WebSSL model from HF Hub: {repo}")
#         model = AutoModel.from_pretrained(repo, revision=revision, trust_remote_code=True)
#         model.eval()
#         return model

#     # Fallback: Load from torch.hub (e.g. custom models with hubconf.py)
#     else:
#         import torch
#         print(f"Loading model via torch.hub: {repo}, {model_name}")
#         model = torch.hub.load(repo, model_name, pretrained=True)
#         model.eval()
#         return model
# ---------  

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
    Supports both standard ViT-style (with or without CLS token) and HuggingFace CLIP-style embeddings.
    No changes if the current grid already matches the target size or is not square.
    """
    new_grid = img_size // patch_size

    # --- DINO / custom ViT style --- #
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
    Load a vision model from HuggingFace, torch.hub, or a local TIPS repo, and resize positional embeddings to match the input size.
    - Automatically handles interpolation and setup for various model types like CLIP, SigLIP, DINO, and TIPS.
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

        # Patch CLIP’s internal guard
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
            from tips.pytorch import image_encoder  # don't forget to add tips to the the path

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

    # Load model
    # model = load_model(args).to(DEVICE) # Old, uses only 1 GPU for evaluation
    model = load_model(args)
    # Wrap in DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs.")
        model = torch.nn.DataParallel(model)
    # Move model to GPU(s)
    model = model.to(DEVICE)


    if not os.path.exists(RESULTS_PATH):
        os.makedirs("results", exist_ok=True)
        with open(RESULTS_PATH, "w") as f:
            f.write("job_id,model,train_bins,val_bin,jac_mean,jac_std,jac0,jac1,jac2,jac3,jac4,jac5,jac6,jac7,jac8,jac9,jac10,jac11,jac12,jac13,jac14,jac15,d_model,batch_size,input_size,patch_size\n")

    # Parse --nn_params if provided, else use empty dict
    if args.nn_params:
        try:
            nn_params = json.loads(args.nn_params)
        except json.JSONDecodeError:
            raise ValueError("Invalid format for --nn_params. Provide a valid JSON string.")
    else:
        nn_params = {}


    # Decide whether to enable FAISS sharding (this might help with out of memory errors)
    num_gpus = torch.cuda.device_count()
    if str(DEVICE) == "cuda" and args.nn_method == "faiss" and num_gpus > 1:
        print(f"Detected {num_gpus} GPUs. Enabling FAISS index sharding.")
        nn_params.setdefault("idx_shard", True)
    else:
        print(f"FAISS sharding not used (device: {DEVICE}, GPUs available: {num_gpus})")
        

    for train_bins in tqdm(TRAIN_BIN_LISTS, mininterval=10):
        hbird_miou = hbird_evaluation(
            model=model,
            d_model=args.d_model,
            patch_size=args.patch_size,
            batch_size=args.batch_size,
            input_size=args.input_size,
            augmentation_epoch=args.augmentation_epoch,
            device=DEVICE,
            return_knn_details=args.return_knn_details,
            nn_method=args.nn_method,
            n_neighbours=args.n_neighbours,
            nn_params=nn_params,
            ftr_extr_fn=token_features,
            dataset_name=args.dataset_name,
            data_dir=f"{args.data_dir}",
            memory_size=args.memory_size,
            num_workers=args.num_workers,
            ignore_index=-1,  # do not ignore any classes
            train_fs_path=args.train_fs_path,
            val_fs_path=args.val_fs_path,
            train_bins=train_bins,
            val_bins=VAL_BINS,
        )

        train_str = '_'.join(str(x) for x in sorted(train_bins))
        
        # The label that will be used in the results file
        model_label = args.model_name or args.model_repo.rstrip("/").split('/')[-1]

        with open(RESULTS_PATH, mode='a', newline='') as file:
            for i in range(len(VAL_BINS)):
                writer = csv.writer(file)
                writer.writerow([
                    JOB_ID, model_label, train_str, VAL_BINS[i], r3(np.mean(hbird_miou[i])),
                    r3(np.std(hbird_miou[i])), *[r3(x) for x in hbird_miou[i]],
                    args.d_model, args.batch_size, args.input_size, args.patch_size
                ])
        print(f"Results saved for train_bins: {train_str}, val_bins: {VAL_BINS}")
    
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

    # MoCo-specific args
    parser.add_argument("--hf_repo", default=None, type=str,
                        help="HuggingFace repo ID for MoCo checkpoint")
    parser.add_argument("--hf_filename", default=None, type=str,
                        help="Checkpoint filename in HuggingFace repo")
    
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
    
    parser.add_argument("--train_bins", type=str, default=None,
                        help="UNUSED")
    parser.add_argument("--val_bins", type=str, default=None,
                        help="UNUSED")
    parser.add_argument("--class_num", type=str, default=None, help="UNUSED")
    
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
    parser.add_argument("--return_knn_details", action="store_true",
                        help="Whether to return details of k-NN results")

    parser.add_argument("--job_id", default=None,
                        help="job_id of job")

    args = parser.parse_args()

    seed_everything(args.seed)

    main(args)
