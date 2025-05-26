import os
import json
import torch
import random
import argparse
import numpy as np
import csv

from hbird.hbird_eval import hbird_evaluation

from tqdm import tqdm

RESULTS_PATH = 'results/results_test.csv'
JOB_ID = os.environ.get('SLURM_JOB_ID')
VAL_BINS = [0, 15, 30, 45, 60, 75, 90]
TRAIN_BIN_LISTS = [
    # [0, 30, 60, 90],
    # [0, 45, 90],
    # [0, 90],
    [0],
]
# Setup device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int):
    """Ensure reproducibility across torch, numpy, and python."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_model(args):
    """
    Load a vision model based on the source.

    - CLIP, SigLIP, RADIO, WebSSL: from Hugging Face (transformers)
    - DINOv2: from torch.hub
    - DINO: from torch.hub
    - TIPS: from local repo (requires cloning the TIPS repo)
    """
    repo = args.model_repo
    revision = getattr(args, "revision", None)
    model_name = getattr(args, "model_name", None)

    # Load CLIP (ViT encoder + projection head)
    if "clip" in repo.lower():
        from transformers import CLIPModel
        print(f"Loading CLIP model from HF Hub: {repo}")
        model = CLIPModel.from_pretrained(repo, revision=revision, trust_remote_code=True)
        model.eval()
        return model

    # Load SigLIP (ViT + MAP head, no CLS token)
    elif "siglip" in repo.lower():
        from transformers import SiglipModel
        print(f"Loading SigLIP model from HF Hub: {repo}")
        model = SiglipModel.from_pretrained(repo, revision=revision, trust_remote_code=True)
        model.eval()
        return model

    # Load RADIO or other custom transformer models from HF
    elif "radio" in repo.lower():
        from transformers import AutoModel
        print(f"Loading RADIO model from HF Hub: {repo}")
        model = AutoModel.from_pretrained(repo, revision=revision, trust_remote_code=True)
        model.eval()
        return model

    # Load DINOv2 from torch.hub
    elif "dinov2" in repo.lower():
        import torch
        print(f"Loading DINOv2 model via torch.hub: {repo}, {model_name}")
        model = torch.hub.load(repo, model_name, pretrained=True)
        model.eval()
        return model

    # Load original DINO from torch.hub
    elif "dino" in repo.lower():
        import torch
        print(f"Loading original DINO model via torch.hub: {repo}, {model_name}")
        model = torch.hub.load(repo, model_name, pretrained=True)
        model.eval()
        return model
    
    # Load TIPS from local repo (using pre-downloaded .npz checkpoints)
    elif "tips" in repo.lower():
        try:
            import numpy as np
            import torch
            from tips.pytorch import image_encoder

            print(f"Loading TIPS model: {repo}")
            variant_key = repo.lower().split("tips-")[-1]

            # Map model variants to checkpoints and corresponding constructor functions
            checkpoint_map = {
                "s14": ("../tips/pytorch/checkpoints/tips_oss_s14_highres_distilled_vision.npz", image_encoder.vit_small),
                "b14": ("../tips/pytorch/checkpoints/tips_oss_b14_highres_distilled_vision.npz", image_encoder.vit_base),
                "l14": ("../tips/pytorch/checkpoints/tips_oss_l14_highres_distilled_vision.npz", image_encoder.vit_large),
                "g14": ("../tips/pytorch/checkpoints/tips_oss_g14_highres_vision.npz", image_encoder.vit_giant2),
                "so400m14": ("../tips/pytorch/checkpoints/tips_oss_so400m14_highres_largetext_distilled_vision.npz", image_encoder.vit_so400m),
            }

            if variant_key not in checkpoint_map:
                raise ValueError(f"Unknown TIPS variant '{variant_key}'. Expected one of: {list(checkpoint_map.keys())}")

            checkpoint_path, model_fn = checkpoint_map[variant_key]

            # Load the .npz checkpoint and convert to PyTorch tensors
            weights_np = np.load(checkpoint_path, allow_pickle=False)
            weights = {}
            for k, v in weights_np.items():
                weights[k] = torch.tensor(v)

            # Initialize and load the vision transformer
            model = model_fn(
                img_size=args.input_size,
                patch_size=args.patch_size,
                ffn_layer='mlp',
                block_chunks=0,
                init_values=1.0,
                interpolate_antialias=True,
                interpolate_offset=0.0,
            )
            model.load_state_dict(weights)
            model.eval()
            return model

        except ImportError:
            raise ImportError(
                "Could not import TIPS. Ensure the TIPS repo is cloned from "
                "https://github.com/google-deepmind/tips and its path is in PYTHONPATH."
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"TIPS checkpoint not found at {checkpoint_path}. "
                "Make sure you've run `download_checkpoints.sh` in the TIPS repo."
            )

    # Load WebSSL from Hugging Face
    elif "webssl" in repo.lower():
        from transformers import AutoModel
        print(f"Loading WebSSL model from HF Hub: {repo}")
        model = AutoModel.from_pretrained(repo, revision=revision, trust_remote_code=True)
        model.eval()
        return model

    # Fallback: Load from torch.hub (e.g. custom models with hubconf.py)
    else:
        import torch
        print(f"Loading model via torch.hub: {repo}, {model_name}")
        model = torch.hub.load(repo, model_name, pretrained=True)
        model.eval()
        return model
    
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
    model = load_model(args).to(DEVICE)

    if not os.path.exists(RESULTS_PATH):
        os.makedirs("results", exist_ok=True)
        with open(RESULTS_PATH, "w") as f:
            f.write("job_id,model,class_num,train_bins,val_bin,jac0,jac1,jac_mean,d_model,input_size,patch_size\n")

    # Parse --nn_params if provided, else use empty dict
    if args.nn_params:
        try:
            nn_params = json.loads(args.nn_params)
        except json.JSONDecodeError:
            raise ValueError("Invalid format for --nn_params. Provide a valid JSON string.")
    else:
        nn_params = {}

    if args.class_num == "all":
        classes = [7, 8, 19, 46, 57, 60, 70, 99, 100, 113, 125, 126, 152, 166, 196]
    else:
        classes = args.class_num.split(",")

    # for class_num in classes:
    try:
        # for train_bins in tqdm(TRAIN_BIN_LISTS, desc=f"Running experiments for class {class_num}", mininterval=10):

        for train_bins in TRAIN_BIN_LISTS:
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
                ignore_index=-1,
                train_fs_path=args.train_fs_path,
                val_fs_path=args.val_fs_path,
                train_bins=train_bins,
                val_bins=VAL_BINS,
            )

            train_str = '_'.join(str(x) for x in sorted(train_bins))
            
            # The label that will be used in the results file
            model_label = args.model_name or args.model_repo.split('/')[-1]

            for i in range(len(VAL_BINS)):
                with open(RESULTS_PATH, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        JOB_ID, model_label, 0, train_str, VAL_BINS[i], hbird_miou[i][0], 
                        hbird_miou[i][1], np.mean(hbird_miou[i]), args.d_model, args.input_size, args.patch_size])
                    print(f"train_bins: {train_str}, class 0 miou: {round(hbird_miou[i][0], 3)}, class 1 miou: {round(hbird_miou[i][1], 3)}") 
    
    except Exception as e:
        print(f"Exception at {class_num=}, {e=}")
        with open(f"results/exp_a_errors.txt", "a") as f:
            f.write(f"{class_num=}, args={args}\n")

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
                        help="(MVImgNet only) Training angle bins as comma-sep list like 0,15,30")
    parser.add_argument("--val_bins", type=str, default=None,
                        help="(MVImgNet only) Validation angle bins as comma-sep list like 0,15,30")
    parser.add_argument("--class_num", type=str, help="(MVImgNet only) object class ('all' or 70_99_125 or similar)")
    
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
