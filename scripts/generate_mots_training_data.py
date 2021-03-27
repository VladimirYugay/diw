import logging
import sys
from pathlib import Path

import click
import cv2
import numpy as np
from IPython.core import ultratb

import diw
from diw.readers.mots_reader import MOTSReader

sys.excepthook = ultratb.FormattedTB(mode="Verbose", color_scheme="Linux", call_pdb=1)

_logger = logging.getLogger(__name__)


def apply_mask(image, masks):
    """ create masks silhouettes"""
    final_mask = np.zeros((image.shape[0], image.shape[1]))
    for mask in masks:
        final_mask[mask == 1] = 1
    return final_mask


def populate_file(dir):
    """ creates a training file with training ids """
    with open(dir / "train.txt", "w") as f:
        for directory in dir.iterdir():
            if "train.txt" in str(directory):
                continue
            print(str(directory).split("/")[-1], file=f)


@click.command()
@click.option(
    "--data_dir",
    "data_dir",
    default="/home/vy/university/thesis/datasets/MOTS/train",
    type=click.Path(exists=True),
    help="Path to MOTS dataset",
)
@click.option(
    "--output_dir",
    "output_dir",
    default="./data/preprocessed/MOTS_preprocessed",
    type=click.Path(exists=True),
    help="Output path of the dataset",
)
@click.version_option(diw.__version__)
def main(data_dir, output_dir):
    height, width = 128, 416
    config = {"resize_shape": [width, height], "read_boxes": False}
    reader = MOTSReader(data_dir, config)
    for seq_id in reader.sequence_info.keys():
        for i in range(1, reader.sequence_info[seq_id]["length"] - 1):
            print("Processing sequence: {}, frame: {}".format(seq_id, i))
            prev_sample, cur_sample, next_sample = (
                reader.read_sample(seq_id, i - 1),
                reader.read_sample(seq_id, i),
                reader.read_sample(seq_id, i + 1),
            )
            stacked_img = np.concatenate(
                [prev_sample["image"], cur_sample["image"], next_sample["image"]],
                axis=1,
            )
            prev_seg, cur_seg, next_seg = (
                apply_mask(prev_sample["image"], prev_sample["masks"]),
                apply_mask(cur_sample["image"], cur_sample["masks"]),
                apply_mask(next_sample["image"], next_sample["masks"]),
            )
            stacked_seg = np.concatenate((prev_seg, cur_seg, next_seg), axis=1)
            new_folder = Path(output_dir) / str(seq_id + "_" + str(i - 1))
            new_folder.mkdir(parents=True, exist_ok=True)
            img_name, seg_name = str(new_folder / "image.jpg"), str(
                new_folder / "seg.jpg"
            )
            cv2.imwrite(img_name, cv2.cvtColor(stacked_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(seg_name, stacked_seg * 255)
    populate_file(Path(output_dir))


if __name__ == "__main__":
    main()
