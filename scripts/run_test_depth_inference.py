""" Script for running depth inference with the model from the checkpoint """
import logging
import os
import sys
from pathlib import Path

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
    help="path to checkpoint directory",
)
@click.option(
    "--data_dir",
    "data_dir",
    default="./data/test/mots_data",
    type=click.Path(exists=True),
    help="path to reader config file",
)
@click.option(
    "--output_dir",
    "output_dir",
    default="./data/output",
    type=click.Path(exists=True),
    help="path to reader config file",
)
@click.version_option(diw.__version__)
def main(data_dir, checkpoint_dir, output_dir):
    height, width = 128, 416
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # to fix CUDA bug
    inference_model = Model(
        is_training=False, batch_size=1, img_height=height, img_width=width
    )
    checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    vars_to_restore = get_vars_to_save_and_restore(checkpoint)
    saver = tf.train.Saver(vars_to_restore)
    plt.figure()
    with tf.Session() as sess:
        saver.restore(sess, checkpoint)
        for img_path in Path(data_dir).glob("**/*"):
            print("Processing", img_path)
            image = load_image(str(img_path))
            image = resize_img(image, (width, height))
            image = np.array(image)
            image = image[None, ...]
            depth = inference_model.inference_depth(image, sess)
            depth = depth[0, :, :, 0]
            plt.imshow(depth, plt.cm.get_cmap("plasma").reversed())
            img_name = str(img_path).split("/")[-1]
            plt.savefig(output_dir + "/" + img_name.replace(".jpg", ".png"))


if __name__ == "__main__":
    main()
