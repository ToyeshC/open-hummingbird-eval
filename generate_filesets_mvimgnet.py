#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import csv
import math
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import pandas as pd


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
    parser.add_argument("--file_sets_root", type=Path,
                        default=Path("file_sets"))
    parser.add_argument("--angle_bins", type=str,
                        default=",".join(map(str, DEFAULT_CONFIG["angle_bins"])))
    parser.add_argument("--classes", type=str,
                        default=",".join(map(str, sorted(DEFAULT_CONFIG["class_labels"].keys()))))

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
        sorted_images = sorted(
            images.items(),
            key=lambda item: int(item[1].name.split(".")[0])
        )

        angles_list = [0]

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

        return angles_list

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

                angles = compute_angles(images)
                steps = compute_steps(angles)

                max_angle = max(angles)
                mean_step = round(float(pd.Series(steps).mean()), 3) if steps else 0.0
                std_step = round(float(pd.Series(steps).std()), 3) if steps else 0.0

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
                }

        return results

    # -------------------------------------------------------------
    # BIN ASSIGNMENT (mirrors notebook behavior)
    # -------------------------------------------------------------

    def assign_images_to_bins(object_folder, angle_data, angle_bins):

        images_dir = object_folder / "images"
        image_files = sorted(
            images_dir.glob("*.jpg"),
            key=lambda x: int(x.stem)
        )

        assigned = defaultdict(list)
        angles = angle_data["angles"]

        for idx, img_path in enumerate(image_files):
            angle_value = angles[idx]

            closest_bin = min(
                angle_bins,
                key=lambda b: abs(b - angle_value)
            )

            assigned[closest_bin].append(img_path.name)

        return assigned

    # -------------------------------------------------------------
    # MAIN COLLECTION
    # -------------------------------------------------------------

    file_sets_output = args.file_sets_root / "mvimgnet" / "full"
    file_sets_output.mkdir(parents=True, exist_ok=True)

    angle_to_paths = defaultdict(list)

    for class_id in sorted(CONFIG["classes"]):

        class_folder = CONFIG["input_path"] / str(class_id)
        if not class_folder.exists():
            continue

        print(f"Processing class {class_id}")

        csv_path = file_sets_output / f"{class_id}_angles.csv"
        results = compute_and_log_angles(class_folder, csv_path)

        for object_name, angle_data in results.items():

            object_folder = class_folder / object_name

            assigned = assign_images_to_bins(
                object_folder,
                angle_data,
                CONFIG["angle_bins"]
            )

            for bin_angle, image_names in assigned.items():
                for img_name in image_names:

                    rel_path = (
                        Path(str(class_id))
                        / object_name
                        / "images"
                        / img_name
                    )

                    angle_to_paths[bin_angle].append(str(rel_path))

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
