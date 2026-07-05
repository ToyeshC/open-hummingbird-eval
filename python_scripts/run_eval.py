import json
import argparse
import numpy as np
import torch

from hbird.hbird_eval import hbird_evaluation
from hbird.utils.repro import seed_everything
from hbird.utils.feature_extractors import token_features
from hbird.utils.loading_models import load_model


def parse_args():
    parser = argparse.ArgumentParser(description="HummingBird Evaluation")

    parser.add_argument(
        "--seed", default=42, type=int, help="Random seed for reproducibility"
    )

    # Model arguments
    parser.add_argument(
        "--model_repo",
        default=None,
        type=str,
        help="Torch Hub repo or HuggingFace repo ID",
    )
    parser.add_argument(
        "--model_name",
        default=None,
        type=str,
        help="Model name for torch.hub (e.g. dino_vits16)",
    )
    parser.add_argument(
        "--d_model",
        default=None,
        type=int,
        help="Size of the embedding feature vectors",
    )
    parser.add_argument(
        "--revision",
        default=None,
        type=str,
        help="(HuggingFace only) Commit hash, tag, or branch to pin model version",
    )

    # Input & patching
    parser.add_argument(
        "--input_size", default=None, type=int, help="Size of the input image"
    )
    parser.add_argument(
        "--patch_size", default=None, type=int, help="Size of the model patch"
    )

    # Dataset arguments
    parser.add_argument(
        "--data_dir", default=None, type=str, help="Path to the dataset root"
    )
    parser.add_argument(
        "--dataset_name",
        default=None,
        type=str,
        help="Dataset name (e.g. voc, mvimgnet)",
    )
    parser.add_argument(
        "--train_fs_path", default=None, type=str, help="Path to train file list"
    )
    parser.add_argument(
        "--val_fs_path", default=None, type=str, help="Path to validation file list"
    )
    # MVImgNet Dataset specific arguments
    parser.add_argument(
        "--train_bins",
        type=str,
        default=None,
        help="(MVImgNet only) Training angle bins as comma-sep list like 0,15,30",
    )
    parser.add_argument(
        "--val_bins",
        type=str,
        default=None,
        help="(MVImgNet only) Validation angle bins as comma-sep list like 0,15,30",
    )

    # Evaluation behavior
    parser.add_argument(
        "--batch_size", default=64, type=int, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--augmentation_epoch",
        default=1,
        type=int,
        help="Number of augmentation passes over training data",
    )
    parser.add_argument(
        "--memory_size", default=None, type=int, help="Optional memory size cap"
    )
    parser.add_argument(
        "--num_workers", default=None, type=int, help="Num workers for DataLoader"
    )

    # Nearest neighbor search
    parser.add_argument(
        "--n_neighbours",
        default=30,
        type=int,
        help="Number of neighbors to use in k-NN search",
    )
    parser.add_argument(
        "--nn_method",
        default="faiss",
        type=str,
        help="Method for nearest neighbor search",
    )
    parser.add_argument(
        "--nn_params",
        default=None,
        type=str,
        help="JSON string for nearest neighbor parameters",
    )
    parser.add_argument(
        "--return_knn_details",
        action=None,
        help="Whether to return details of k-NN results",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    print("Args:", args)

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args)

    if (
        torch.cuda.device_count() > 1
    ):  # make all GPUs work in parallel on the batch (if more than 1 GPU is available)
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs.")
        model = torch.nn.DataParallel(model)

    model = model.to(device)

    if hasattr(model, "interpolate_pos_encoding"):
        print(
            "The model is loaded from Hugging Face and supports auto embedding interpolation, \n"
            "however manual embedding interpolation will be applied if the input_size provided does not match \n"
            "the input_size on which the model was trained."
        )
    else:
        print(
            "The model does not support automatic embedding interpolation, \n"
            "but manual embedding interpolation will be applied if the input_size provided does not match \n"
            "the input_size on which the model was trained."
        )

    if args.nn_params:
        try:
            nn_params = json.loads(args.nn_params)
        except json.JSONDecodeError:
            raise ValueError(
                "Invalid format for --nn_params. Provide a valid JSON string."
            )
    else:
        nn_params = {}

    # Decide whether to enable FAISS sharding (moves faiss index to multiple GPUs and helps with OOM errors)
    num_gpus = torch.cuda.device_count()
    if str(device) == "cuda" and args.nn_method == "faiss" and num_gpus > 1:
        print(f"Detected {num_gpus} GPUs. Enabling FAISS index sharding.")
        nn_params.setdefault("idx_shard", True)
    else:
        print(f"FAISS sharding not used (device: {device}, GPUs available: {num_gpus})")

    def token_features_fn(model, imgs):
        """
        Wrapper function for passing `args` to `token_features`.
        """
        return token_features(args, model, imgs)

    if args.dataset_name.lower() == "mvimgnet":
        # Handle MVImgNet angle bin parsing

        assert args.train_bins is not None, (
            "You must specify --train_bins for mvimgnet."
        )
        assert args.val_bins is not None, "You must specify --val_bins for mvimgnet."

        train_bins_list = args.train_bins.split(",")
        val_bins_list = args.val_bins.split(",")

        assert type(train_bins_list) is list, (
            "train_bins must be a comma-separated list."
        )
        assert type(val_bins_list) is list, "val_bins must be a comma-separated list."

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
            ftr_extr_fn=token_features_fn,
            dataset_name=args.dataset_name,
            data_dir=args.data_dir,
            memory_size=args.memory_size,
            num_workers=args.num_workers,
            train_fs_path=args.train_fs_path,
            val_fs_path=args.val_fs_path,
            train_bins=train_bins_list,
            val_bins=val_bins_list,
            sequence_length=1,
        )
        print(
            f"val_bin(s): {val_bins_list}, Hummingbird Evaluation (mIoU): {np.mean(hbird_miou_list)}"
        )
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
            ftr_extr_fn=token_features_fn,
            dataset_name=args.dataset_name,
            data_dir=args.data_dir,
            memory_size=args.memory_size,
            num_workers=args.num_workers,
            train_fs_path=args.train_fs_path,
            val_fs_path=args.val_fs_path,
            sequence_length=1,
        )
        print(f"Hummingbird Evaluation (mIoU): {hbird_miou}")


if __name__ == "__main__":
    main()
