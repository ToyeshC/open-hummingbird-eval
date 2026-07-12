#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import sys
import csv
import math
from pathlib import Path
from collections import defaultdict
from datetime import datetime


# ---------------------------------------------------------------------
# DEFAULT CONFIG (copied from notebook)
# ---------------------------------------------------------------------

DEFAULT_CONFIG = {
    "colmap_script": Path("../colmap/scripts/python"),
    "input_path": Path("../datasets/mvimgnet"),
    "angle_bins": [0, 15, 30, 45, 60, 75, 90],
    "class_labels": {
        7: "Stove", 8: "Sofa", 19: "Microwave", 46: "Bed",
        57: "Toy Cat", 60: "Toy Cow", 70: "Toy Dragon",
        99: "Coat Rack", 100: "Guitar Stand", 113: "Ceiling Lamp",
        125: "Toilet", 126: "Sink", 152: "Strings",
        166: "Broccoli", 196: "Durian"
    },
    # Expected images per class per angle bin for the paper's 15-class MVImgNet
    # subset (one frame per kept capture per bin, so identical across bins).
    # A mismatch means the raw data or masks are incomplete.
    "class_n_images": {
        7: 197, 8: 91, 19: 120, 46: 23, 57: 783, 60: 735, 70: 627,
        99: 97, 100: 218, 113: 154, 125: 58, 126: 30, 152: 192,
        166: 210, 196: 758
    },
}


# ---------------------------------------------------------------------
# ARGUMENT PARSING
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate file_sets/mvimgnet/full/angle_X.txt using exact notebook logic."
    )

    parser.add_argument("--colmap_script", type=Path,
                        default=DEFAULT_CONFIG["colmap_script"])
    parser.add_argument("--input_path", type=Path,
                        default=DEFAULT_CONFIG["input_path"])
    parser.add_argument("--masks_path", type=Path, default=None,
                        help="Masks root (<masks>/<class>/<capture>/<frame>.jpg.png); "
                             "if given, only frames with masks are assigned to bins")
    parser.add_argument("--file_sets_root", type=Path,
                        default=Path("file_sets"))
    parser.add_argument("--angle_bins", type=str,
                        default=",".join(map(str, DEFAULT_CONFIG["angle_bins"])))
    parser.add_argument("--classes", type=str,
                        default=",".join(map(str, sorted(DEFAULT_CONFIG["class_labels"].keys()))))
    parser.add_argument("--skip_count_check", action="store_true",
                        help="Skip verifying per-class per-bin counts against the known "
                             "totals of the paper's 15-class subset (use for custom subsets)")

    return parser.parse_args()


def parse_csv_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main() -> int:

    args = parse_args()

    CONFIG = {
        "colmap_script": args.colmap_script,
        "input_path": args.input_path,
        "angle_bins": parse_csv_int_list(args.angle_bins),
        "classes": parse_csv_int_list(args.classes),
    }

    # -------------------------------------------------------------
    # EXACT NOTEBOOK COLMAP IMPORT
    # -------------------------------------------------------------

    assert (CONFIG["colmap_script"] / "read_write_model.py").exists(), \
        f"Missing: {CONFIG['colmap_script']}/read_write_model.py"

    sys.path.append(str(CONFIG["colmap_script"]))
    from read_write_model import read_images_binary, qvec2rotmat

    print("OK: imported read_images_binary, qvec2rotmat")

    # -------------------------------------------------------------
    # COPIED: compute_angles (exact logic)
    # -------------------------------------------------------------

    def compute_angles(images):
        """Return both the angle list (notebook-compatible) and a name->angle map.

        The map keys on the COLMAP-registered image name so bin assignment cannot
        drift when frames on disk were not registered by COLMAP.
        """
        sorted_images = sorted(
            images.items(),
            key=lambda item: int(item[1].name.split(".")[0])
        )

        angles_list = [0]
        name_to_angle = {Path(sorted_images[0][1].name).name: 0.0}

        for i in range(1, len(sorted_images)):
            prev = sorted_images[0][0]
            curr = sorted_images[i][0]

            image1 = images[prev]
            image2 = images[curr]

            R1 = qvec2rotmat(image1.qvec)
            R2 = qvec2rotmat(image2.qvec)

            R_rel = R2 @ R1.T
            trace = R_rel.trace()

            angle = (180.0 / math.pi) * math.acos(
                max(min((trace - 1) / 2, 1), -1)
            )

            angles_list.append(round(angle, 3))
            name_to_angle[Path(sorted_images[i][1].name).name] = round(angle, 3)

        return angles_list, name_to_angle

    # -------------------------------------------------------------
    # COPIED: compute_steps (exact logic)
    # -------------------------------------------------------------

    def compute_steps(angles_list):
        steps_list = []
        for i in range(1, len(angles_list)):
            prev_angle = angles_list[i - 1]
            curr_angle = angles_list[i]
            steps_list.append(round(abs(prev_angle - curr_angle), 3))
        return steps_list

    # -------------------------------------------------------------
    # COPIED: compute_and_log_angles (CSV identical behavior)
    # -------------------------------------------------------------

    def compute_and_log_angles(class_folder_path, csv_path):

        results = {}

        with open(csv_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([
                "folder",
                "num_images",
                "angles",
                "steps",
                "max_angle",
                "mean_step",
                "std_step"
            ])

            for object_folder in sorted(class_folder_path.iterdir()):
                if not object_folder.is_dir():
                    continue

                sparse_path = object_folder / "sparse" / "0" / "images.bin"
                if not sparse_path.exists():
                    continue

                images = read_images_binary(str(sparse_path))
                if len(images) < 2:
                    continue

                angles, name_to_angle = compute_angles(images)
                steps = compute_steps(angles)

                max_angle = max(angles)
                # statistics.stdev matches pandas' sample std (ddof=1)
                mean_step = round(statistics.fmean(steps), 3) if steps else 0.0
                std_step = round(statistics.stdev(steps), 3) if len(steps) >= 2 else 0.0

                writer.writerow([
                    object_folder.name,
                    len(images),
                    angles,
                    steps,
                    max_angle,
                    mean_step,
                    std_step
                ])

                results[object_folder.name] = {
                    "angles": angles,
                    "max_angle": max_angle,
                    "name_to_angle": name_to_angle,
                }

        return results

    # -------------------------------------------------------------
    # BIN ASSIGNMENT (mirrors notebook behavior)
    # -------------------------------------------------------------

    def assign_images_to_bins(object_folder, angle_data, angle_bins, masks_root=None, class_id=None):
        """Mirror the mvimgnet_create_bins notebook: for each target bin pick the
        single closest frame of the capture (by COLMAP-registered name, so frames
        without a pose are never assigned). If masks_root is given, only frames
        whose mask exists are eligible (the notebook links image/mask pairs).

        Also returns the number of frames excluded because their mask is missing,
        so an incomplete masks upload/unzip is visible instead of silently
        changing which frames get selected."""

        images_dir = object_folder / "images"
        on_disk = {p.name for p in images_dir.glob("*.jpg")}

        candidates = {}
        n_missing_mask = 0
        for name, angle_value in angle_data["name_to_angle"].items():
            if name not in on_disk:
                continue
            if masks_root is not None:
                mask_path = masks_root / str(class_id) / object_folder.name / f"{name}.png"
                if not mask_path.is_file():
                    n_missing_mask += 1
                    continue
            candidates[name] = angle_value

        assigned = defaultdict(list)
        if not candidates:
            return assigned, n_missing_mask

        for bin_angle in angle_bins:
            closest_name = min(
                candidates,
                key=lambda n: abs(candidates[n] - bin_angle)
            )
            assigned[bin_angle].append(closest_name)

        return assigned, n_missing_mask

    # -------------------------------------------------------------
    # MAIN COLLECTION
    # -------------------------------------------------------------

    if args.masks_path is not None and not args.masks_path.is_dir():
        print(f"ERROR: --masks_path does not exist: {args.masks_path}")
        return 1

    file_sets_output = args.file_sets_root / "mvimgnet" / "full"
    file_sets_output.mkdir(parents=True, exist_ok=True)

    angle_to_paths = defaultdict(list)
    total_missing_masks = 0
    total_dropped_captures = 0

    for class_id in sorted(CONFIG["classes"]):

        class_folder = CONFIG["input_path"] / str(class_id)
        if not class_folder.exists():
            continue

        print(f"Processing class {class_id}")

        csv_path = file_sets_output / f"{class_id}_angles.csv"
        results = compute_and_log_angles(class_folder, csv_path)

        class_kept = 0
        class_missing_masks = 0
        class_dropped_captures = 0

        for object_name, angle_data in results.items():

            # The notebook keeps only captures that span the full bin range
            if angle_data["max_angle"] < max(CONFIG["angle_bins"]):
                continue

            object_folder = class_folder / object_name

            assigned, n_missing_mask = assign_images_to_bins(
                object_folder,
                angle_data,
                CONFIG["angle_bins"],
                masks_root=args.masks_path,
                class_id=class_id,
            )
            class_missing_masks += n_missing_mask
            if not assigned:
                class_dropped_captures += 1
                continue
            class_kept += 1

            for bin_angle, image_names in assigned.items():
                for img_name in image_names:

                    rel_path = (
                        Path(str(class_id))
                        / object_name
                        / "images"
                        / img_name
                    )

                    angle_to_paths[bin_angle].append(str(rel_path))

        print(f"  class {class_id}: {class_kept} captures kept"
              + (f", {class_missing_masks} frames excluded (missing masks)"
                 f", {class_dropped_captures} captures dropped (no usable frames)"
                 if args.masks_path is not None else ""))
        total_missing_masks += class_missing_masks
        total_dropped_captures += class_dropped_captures

    if total_missing_masks or total_dropped_captures:
        print(f"WARNING: {total_missing_masks} frames excluded due to missing masks and "
              f"{total_dropped_captures} captures dropped entirely (under {args.masks_path}). "
              f"If the masks upload/unzip was interrupted, complete it and regenerate the file sets.")

    # -------------------------------------------------------------
    # VERIFY COUNTS (mirrors the create_bins notebook's verification cell)
    # -------------------------------------------------------------

    expected_counts = DEFAULT_CONFIG["class_n_images"]
    if not args.skip_count_check:
        mismatches = []
        for angle in CONFIG["angle_bins"]:
            per_class = defaultdict(int)
            for rel_path in angle_to_paths[angle]:
                per_class[int(rel_path.split("/", 1)[0])] += 1
            for class_id in CONFIG["classes"]:
                expected = expected_counts.get(class_id)
                if expected is not None and per_class[class_id] != expected:
                    mismatches.append(
                        f"class {class_id} bin {angle}: {per_class[class_id]} != expected {expected}")
        if mismatches:
            for m in mismatches:
                print(f"COUNT MISMATCH: {m}")
            print("Raw data or masks are likely incomplete. Fix the dataset and regenerate, "
                  "or pass --skip_count_check for a custom subset.")
            return 1
        checked = [c for c in CONFIG["classes"] if c in expected_counts]
        if checked:
            print(f"Count check OK: {len(checked)} classes x {len(CONFIG['angle_bins'])} bins "
                  f"match the expected per-class totals.")

    # -------------------------------------------------------------
    # WRITE TXT FILES
    # -------------------------------------------------------------

    for angle in CONFIG["angle_bins"]:

        txt_path = file_sets_output / f"angle_{angle}.txt"

        with open(txt_path, "w") as f:
            for rel_path in angle_to_paths[angle]:
                f.write(rel_path + "\n")

        print(f"Wrote {len(angle_to_paths[angle])} paths to {txt_path}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
