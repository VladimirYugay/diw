#!/usr/bin/env python
import io
import logging
import os
import random
import sys

import click
import numpy as np
import tensorflow.compat.v1 as tf

import diw
from diw.model import Model

_logger = logging.getLogger(__name__)
gfile = tf.gfile
MAX_TO_KEEP = 1000000  # Maximum number of checkpoints to keep.


def load(filename):
    with gfile.Open(filename) as f:
        return np.load(io.BytesIO(f.read()))


def _print_losses(dir1):
    for f in gfile.ListDirectory(dir1):
        if "loss" in f:
            print("----------", f, end=" ")
            f1 = os.path.join(dir1, f)
            t1 = load(f1).astype(float)
            print(t1)


@click.command()
@click.argument("file_extension", default="jpg")
@click.argument("learning_rate", default=1e-4)
@click.argument("reconstr_weight", default=0.85)
@click.argument("ssim_weight", default=3.0)
@click.argument("smooth_weight", default=1e-2)
@click.argument("depth_consistency_loss_weight", default=0.01)
@click.argument("batch_size", default=4)
@click.argument("img_height", default=128)
@click.argument("img_width", default=416)
@click.argument("queue_size", default=2000)
@click.argument("seed", default=8964)
@click.argument("weight_reg", default=1e-2)
@click.argument("train_steps", default=int(1e6))
@click.argument("summary_freq", default=100)
@click.argument("save_freq", default=100)
@click.argument("save_intrinsics", default=100)
@click.argument("input_file", default="train")
@click.argument("rotation_consistency_weight", default=1e-3)
@click.argument("translation_consistency_weight", default=1e-2)
@click.argument("foreground_dilation", default=8)
@click.argument("learn_intrinsics", default=True)
@click.argument("boxify", default=True)
@click.option(
    "--data_dir",
    "data_dir",
    default="/home/vy/university/thesis/datasets/MOTSChallenge_PREPROCESSED/",
    type=click.Path(exists=True),
    help="path to reader config file",
)
@click.option(
    "--checkpoint_dir",
    "checkpoint_dir",
    default="./data/checkpoints/test",
    type=click.Path(exists=True),
    help="path to checkpoint directory",
)
@click.option(
    "--imagenet_ckpt",
    "imagenet_ckpt",
    type=click.Path(exists=True),
    help="path to imagenet checkpoint directory",
)
@click.version_option(diw.__version__)
def main(
    file_extension,
    learning_rate,
    reconstr_weight,
    ssim_weight,
    smooth_weight,
    depth_consistency_loss_weight,
    batch_size,
    img_height,
    img_width,
    queue_size,
    seed,
    weight_reg,
    train_steps,
    summary_freq,
    save_freq,
    save_intrinsics,
    input_file,
    rotation_consistency_weight,
    translation_consistency_weight,
    foreground_dilation,
    learn_intrinsics,
    boxify,
    imagenet_ckpt,
    data_dir,
    checkpoint_dir,
):
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    Model(
        boxify=boxify,
        data_dir=data_dir,
        file_extension=file_extension,
        is_training=True,
        foreground_dilation=foreground_dilation,
        learn_intrinsics=learn_intrinsics,
        learning_rate=learning_rate,
        reconstr_weight=reconstr_weight,
        smooth_weight=smooth_weight,
        ssim_weight=ssim_weight,
        translation_consistency_weight=translation_consistency_weight,
        rotation_consistency_weight=rotation_consistency_weight,
        batch_size=batch_size,
        img_height=img_height,
        img_width=img_width,
        weight_reg=weight_reg,
        depth_consistency_loss_weight=depth_consistency_loss_weight,
        queue_size=queue_size,
        input_file=input_file,
    )


if __name__ == "__main__":
    main()
