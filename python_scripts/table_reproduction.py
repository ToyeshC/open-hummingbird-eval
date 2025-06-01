import os
import json
import torch
import random
import argparse
import numpy as np
import csv

from hbird.hbird_eval import hbird_evaluation


def seed_everything(seed: int):
    """Ensure reproducibility across torch, numpy, and python."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_model(args):
    """
    Load model based on the source:
    - DINO from torch.hub
    - MoCo from HuggingFace (ResNet-50 backbone)
    - Transformers from HuggingFace
    """
    repo = args.model_repo

    if '/' in repo and not repo.startswith('facebookresearch/'):
        from transformers import AutoModel
        print(f"Loading transformer model from HF Hub: {repo}")
        model = AutoModel.from_pretrained(repo, trust_remote_code=True)
        model.eval()
        return model
    
    if 'moco' in repo.lower():
        from torchvision.models import resnet50
        from huggingface_hub import hf_hub_download

        model = resnet50(pretrained=False)

        ckpt_path = hf_hub_download(
            repo_id=args.hf_repo,
            filename=args.hf_filename,
            cache_dir=os.path.expanduser('~/.cache/moco'),
            local_dir_use_symlinks=False,
        )
        print(f"Loaded MoCo checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)
        cleaned = {k.replace('module.encoder_q.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(cleaned, strict=True)
        model.eval()
        return model
    
    if 'clip' in args.model_repo.lower():
        from transformers import CLIPModel
        print(f"Loading CLIP model from HF Hub: {repo}")
        model = CLIPModel.from_pretrained(repo)
        model.eval()
        return model

    # Default: DINO via torch.hub
    print(f"Loading model via torch.hub: {repo}, {args.model_name}")
    model = torch.hub.load(repo, args.model_name, pretrained=True)
    model.eval()
    return model

def main(args):
    print(f"The script arguments are {args}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args).to(device)
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

    # Define feature extractor hook
    def token_features(model, imgs):
        if "dinov2" in args.model_repo.lower():
            return model.forward_features(imgs)['x_norm_patchtokens'], None
        else:  # For DINO and possibly other models
            return model.get_intermediate_layers(imgs)[0][:, 1:], None  # CLS token excluded

    # Run Hummingbird evaluation
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

    # PASCAL
    hbird_miou = hbird_evaluation(model.to(device),
        d_model=args.d_model,        # size of the embedding feature vectors of patches
        patch_size=args.patch_size,
        batch_size = args.batch_size,
        input_size=args.input_size,
        augmentation_epoch=1,     # how many iterations of augmentations to use on top of the training dataset in order to generate the memory
        device=device,
        return_knn_details=False, # whether to return additional NNs details
        nn_method='faiss',
        n_neighbours=30,         # the number of neighbors to fetch per image patch
        nn_params=None,           # Other parameters to be used for the k-NN operator
        ftr_extr_fn=token_features,           # function that extracts features from a vision encoder on images
        dataset_name=args.dataset_name,       # the name of the dataset to use, currently only Pascal VOC is included.
        data_dir=args.data_dir,    # path to the dataset to use for evaluation
        memory_size=args.memory_size,
        train_fs_path=args.train_fs_path,
        val_fs_path=args.val_fs_path,
        num_workers=2,  # Google Colab says it can only support 2 workers. This fixes a warning and a session crash. The default is 8.
        )
    print(f"val_bin(s) :,  Hummingbird Evaluation (mIoU): {hbird_miou}")


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

    # Input & patching
    parser.add_argument("--input_size", default=None, type=int, help="Size of the input image")
    parser.add_argument("--patch_size", default=None, type=int, help="Size of the model patch")

    # Dataset arguments
    parser.add_argument("--data_dir", default=None, type=str,
                        help="Path to the dataset root")
    parser.add_argument("--dataset_name", default=None, type=str, help="Dataset name (e.g. voc)")
    parser.add_argument("--train_fs_path", default=None, type=str,
                        help="Path to train file list")
    parser.add_argument("--val_fs_path", default=None, type=str,
                        help="Path to validation file list")
    
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

    args = parser.parse_args()

    seed_everything(args.seed)

    main(args)
