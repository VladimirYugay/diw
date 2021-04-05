""" Script for running depth inference assuming MOTS dataset structure """
import logging
import os
import sys
from pathlib import Path, PurePath

import click
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
from IPython.core import ultratb
from PIL import Image

import diw
from diw.model import Model, get_vars_to_save_and_restore

sys.excepthook = ultratb.FormattedTB(mode="Verbose", color_scheme="Linux", call_pdb=1)

_logger = logging.getLogger(__name__)


def load_image(img_file):
    """Load image from disk. Output value range: [0,255]."""
    return Image.open(img_file).convert("RGB")


def resize_img(img, img_shape):
    """ resizes an image """
    return img.resize(img_shape, Image.LANCZOS).convert("RGB")


def plot_image(image, image_type="RGB"):
    """ plots image with matplotlib """
    plt.figure()
    color_map = None
    if image_type != "RGB":
        color_map = plt.cm.get_cmap("plasma").reversed()
    plt.imshow(image, cmap=color_map)
    plt.show()  # display it
    return plt


@click.command()
@click.option(
    "--checkpoint_dir",
    "checkpoint_dir",
    default="./data/checkpoints/test",
    type=click.Path(exists=True),
    help="Path to the model checkpoint",
)
@click.option(
    "--data_dir",
    "data_dir",
    default="./data/test/mots_data",
    type=click.Path(exists=True),
    help="Path to MOTS data",
)
@click.version_option(diw.__version__)
def main(data_dir, checkpoint_dir):
    height, width = 128, 416
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # to fix CUDA bug
    inference_model = Model(
        is_training=False, batch_size=1, img_height=height, img_width=width
    )
    checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    vars_to_restore = get_vars_to_save_and_restore(checkpoint)
    saver = tf.train.Saver(vars_to_restore)
    with tf.Session() as sess:
        saver.restore(sess, checkpoint)
        sequence_paths = sorted([p for p in Path(data_dir).glob("*") if p.is_dir()])
        for seq_path in sequence_paths:
            rotations, translations = [], []
            model_name = PurePath(checkpoint_dir).parts[-1] + "_egomotion"
            (seq_path / model_name).mkdir(parents=True, exist_ok=True)
            img_paths = sorted(
                [p for p in (seq_path / "img1").glob("*") if p.is_file()],
                key=lambda path: str(path),
            )
            for i in range(len(img_paths) - 1):
                print("Processing sequence {}, frame {}".format(seq_path, i + 1))
                left_img_path = img_paths[i]
                left_img = load_image(str(left_img_path))
                left_img = resize_img(left_img, (width, height))
                left_img = np.array(left_img)
                left_img = left_img[None, ...]

                right_img_path = img_paths[i + 1]
                right_img = load_image(str(right_img_path))
                right_img = resize_img(right_img, (width, height))
                right_img = np.array(right_img)
                right_img = right_img[None, ...]

                rotation, translation = inference_model.inference_egomotion(
                    left_img, right_img, sess
                )
                rotations.append(rotation[0, ...])
                translations.append(translation[0, ...])
            rotations = np.array(rotations)
            translations = np.array(translations)
            np.save(str(seq_path / model_name / "rotations.npy"), rotations)
            np.save(str(seq_path / model_name / "translations.npy"), translations)


if __name__ == "__main__":
    main()
