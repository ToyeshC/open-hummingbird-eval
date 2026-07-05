import os
import json
import argparse
import numpy as np
import csv
import torch
from tqdm import tqdm

from hbird.hbird_eval import hbird_evaluation
from hbird.utils.repro import seed_everything
from hbird.utils.feature_extractors import token_features
from hbird.utils.loading_models import load_model
from hbird.utils.feature_extractors import load_vggt, VGGTFeatureExtractor

# RESULTS_PATH = 'results/results_exp_a_500_sharding_batch4_workers8_dataparallel_memory10240000_new.csv'  # this is the original memory size used in the paper
# RESULTS_PATH = 'results/results_exp_a_500_sharding_batch4_workers8_dataparallel_memory1024000_new.csv'
RESULTS_PATH = "results/results_exp_a_500_sharding_batch4_workers8_dataparallel_new.csv"
JOB_ID = os.environ.get("SLURM_JOB_ID")
VAL_BINS = [0, 15, 30, 45, 60, 75, 90]
TRAIN_BIN_LISTS = [
    [0, 30, 60, 90],
    [0, 45, 90],
    [0, 90],
    [0],
]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def r3(x, to=3):
    """
    Round x to 3 (default) decimals.
    """
    return round(x, to)


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
        default=False,
        type=bool,
        help="Whether to return details of k-NN results",
    )

    parser.add_argument("--job_id", default=None, help="job_id of job")

    # VGGT-specific arguments
    parser.add_argument(
        "--vggt_hf_id",
        default=None,
        type=str,
        help="Hugging Face model id for VGGT (e.g. facebook/VGGT-1B)",
    )
    parser.add_argument(
        "--vggt_ckpt",
        default=None,
        type=str,
        help="Optional path to a local VGGT checkpoint (.pt/.pth)",
    )
    parser.add_argument(
        "--vggt_normalize",
        action="store_true",
        help="Apply VGGT-specific normalization if required",
    )
    parser.add_argument(
        "--vggt_seq_len",
        default=1,
        type=int,
        help="Number of frames per sample when using VGGT",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    print("Args:", args)

    seed_everything(args.seed)
    feature_extractor = None

    if args.model_repo and "vggt" in args.model_repo.lower():
        print(f"Loading VGGT model via VGGT loader: repo={args.model_repo}, name={args.model_name}")
        backbone = args.model_name or "vggt-1b"
        hf_id = args.vggt_hf_id or args.model_repo or "facebook/VGGT-1B"
        if args.patch_size is None:
            args.patch_size = 14
        else:
            assert args.patch_size == 14, "VGGT checkpoint expects patch size 14"
        if args.vggt_seq_len < 1:
            raise ValueError("--vggt_seq_len must be >= 1")
        vggt_model = load_vggt(
            backbone=backbone,
            ckpt_path=args.vggt_ckpt,
            hf_model_id=hf_id,
            device=DEVICE,
        )
        eval_spatial_resolution = args.input_size // args.patch_size
        feature_extractor = VGGTFeatureExtractor(
            vggt_model,
            eval_spatial_resolution=eval_spatial_resolution,
            d_model=None,
            normalize=args.vggt_normalize,
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 3, args.input_size, args.input_size, device=DEVICE)
            feature_extractor.forward_features(dummy)
        if args.d_model is None:
            args.d_model = feature_extractor.d_model
        print(f"[VGGT] Detected patch embedding dimension: {args.d_model}")
        model = vggt_model.to(DEVICE)
    else:
        model = load_model(args)
        if (
            torch.cuda.device_count() > 1
        ):  # make all GPUs work in parallel on the batch (if more than 1 GPU is available)
            print(f"Using DataParallel with {torch.cuda.device_count()} GPUs.")
            model = torch.nn.DataParallel(model)
        model = model.to(DEVICE)  # move model to GPU(s)

    if not os.path.exists(RESULTS_PATH):
        os.makedirs("results", exist_ok=True)
        with open(RESULTS_PATH, "w") as f:
            f.write(
                "job_id,model,train_bins,val_bin,jac_mean,jac_std,jac0,jac1,jac2,jac3,jac4,jac5,jac6,jac7,jac8,jac9,jac10,jac11,jac12,jac13,jac14,jac15,d_model,batch_size,input_size,patch_size\n"
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
    if str(DEVICE) == "cuda" and args.nn_method == "faiss" and num_gpus > 1:
        print(f"Detected {num_gpus} GPUs. Enabling FAISS index sharding.")
        nn_params.setdefault("idx_shard", True)
    else:
        print(f"FAISS sharding not used (device: {DEVICE}, GPUs available: {num_gpus})")

    def token_features_fn(model, imgs):
        return token_features(args, model, imgs)

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
            ftr_extr_fn=None if feature_extractor is not None else token_features_fn,
            dataset_name=args.dataset_name,
            data_dir=f"{args.data_dir}",
            memory_size=args.memory_size,
            num_workers=args.num_workers,
            train_fs_path=args.train_fs_path,
            val_fs_path=args.val_fs_path,
            train_bins=train_bins,
            val_bins=VAL_BINS,
            feature_extractor=feature_extractor,
            sequence_length=args.vggt_seq_len if feature_extractor is not None else 1,
        )

        train_str = "_".join(str(x) for x in sorted(train_bins))

        # The label that will be used in the results file
        model_label = args.model_name or args.model_repo.rstrip("/").split("/")[-1]

        with open(RESULTS_PATH, mode="a", newline="") as file:
            for i in range(len(VAL_BINS)):
                writer = csv.writer(file)
                writer.writerow(
                    [
                        JOB_ID,
                        model_label,
                        train_str,
                        VAL_BINS[i],
                        r3(np.mean(hbird_miou[i])),
                        r3(np.std(hbird_miou[i])),
                        *[r3(x) for x in hbird_miou[i]],
                        args.d_model,
                        args.batch_size,
                        args.input_size,
                        args.patch_size,
                    ]
                )
        print(f"Results saved for train_bins: {train_str}, val_bins: {VAL_BINS}")


if __name__ == "__main__":
    main()
