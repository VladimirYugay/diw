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
    help="Path to KITTI data",
)
@click.option(
    "--save_img",
    "save_img",
    flag_value=True,
    help="Flag to whether save the image of the depth (besides numpy array)",
)
@click.version_option(diw.__version__)
def main(data_dir, checkpoint_dir, save_img):
    if save_img:
        plt.figure()
    height, width = 128, 416
    data_dir = Path(data_dir)

    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # to fix CUDA bug
    inference_model = Model(
        is_training=False, batch_size=1, img_height=height, img_width=width
    )
    checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    vars_to_restore = get_vars_to_save_and_restore(checkpoint)
    saver = tf.train.Saver(vars_to_restore)
    with tf.Session() as sess:
        saver.restore(sess, checkpoint)
        sequence_paths = sorted(
            [p for p in (data_dir / "image_02").glob("*") if p.is_dir()],
            key=lambda p: str(p),
        )
        for seq_path in sequence_paths:
            seq_name = Path(seq_path.parts[-1])
            model_name = PurePath(checkpoint_dir).parts[-1]
            (data_dir / model_name).mkdir(parents=True, exist_ok=True)
            (data_dir / model_name / seq_name).mkdir(parents=True, exist_ok=True)
            if save_img:
                (data_dir / (model_name + "_depth_images") / seq_name).mkdir(
                    parents=True, exist_ok=True
                )

            img_paths = sorted(
                [
                    p
                    for p in (data_dir / "image_02" / seq_name).glob("*")
                    if p.is_file()
                ],
                key=lambda path: str(path),
            )

            for img_path in img_paths:
                img_name = img_path.parts[-1].split(".")[0]
                print("Processing sequence: {}, image: {}".format(seq_path, img_name))
                image = load_image(str(img_path))
                image = resize_img(image, (width, height))
                image = np.array(image)
                image = image[None, ...]
                depth = inference_model.inference_depth(image, sess)
                depth = depth[0, :, :, 0]
                np.save(str(data_dir / model_name / seq_name / img_name), depth)
                if save_img:
                    plt.imshow(depth, plt.cm.get_cmap("plasma").reversed())
                    plt.savefig(
                        str(data_dir / (model_name + "_depth_images") / seq_name)
                        + "/"
                        + (img_name + ".png")
                    )
                    plt.clf()


if __name__ == "__main__":
    main()
