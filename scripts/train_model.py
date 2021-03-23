#!/usr/bin/env python
import io
import logging
import math
import os
import random
import sys
import time

import click
import numpy as np
import tensorflow.compat.v1 as tf
from IPython.core import ultratb

import diw
from diw.model import Model

sys.excepthook = ultratb.FormattedTB(mode="Verbose", color_scheme="Linux", call_pdb=1)

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
@click.argument("batch_size", default=4)
@click.argument("depth_consistency_loss_weight", default=0.01)
@click.argument("file_extension", default="jpg")
@click.argument("foreground_dilation", default=8)
@click.argument("img_height", default=128)
@click.argument("img_width", default=416)
@click.argument("input_file", default="train")
@click.argument("learning_rate", default=1e-4)
@click.argument("queue_size", default=2000)
@click.argument("reconstr_weight", default=0.85)
@click.argument("rotation_consistency_weight", default=1e-3)
@click.argument("ssim_weight", default=3.0)
@click.argument("smooth_weight", default=1e-2)
@click.argument("seed", default=8964)
@click.argument("summary_freq", default=100)
@click.argument("save_freq", default=100)
@click.argument("save_intrinsics", default=100)
@click.argument("train_steps", default=23840)
@click.argument("translation_consistency_weight", default=1e-2)
@click.argument("weight_reg", default=1e-2)
@click.option(
    "--boxify",
    "boxify",
    help="Turn masks into boxes",
    flag_value=True,
)
@click.option(
    "--checkpoint_dir",
    "checkpoint_dir",
    default="./data/checkpoints/test",
    help="path to checkpoint directory",
)
@click.option(
    "--data_dir",
    "data_dir",
    default="/home/vy/university/thesis/datasets/MOTSChallenge_PREPROCESSED/",
    type=click.Path(exists=True),
    help="path to reader config file",
)
@click.option(
    "--debug",
    "debug",
    help="One training step performed and results dumped into deubg folder",
    flag_value=True,
)
@click.option(
    "--imagenet_ckpt",
    "imagenet_ckpt",
    type=click.Path(exists=True),
    help="path to imagenet checkpoint directory",
)
@click.option(
    "--learn_intrinsics",
    "learn_intrinsics",
    help="Learn intrinsics",
    flag_value=True,
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
    debug,
):
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if not gfile.Exists(checkpoint_dir):
        gfile.MakeDirs(checkpoint_dir)

    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_model = Model(
        boxify=boxify,
        data_dir=data_dir,
        file_extension=file_extension,
        is_training=True,
        foreground_dilation=foreground_dilation,
        learn_intrinsics=True,  # set up for true meanwhile
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

    saver = train_model.saver
    sv = tf.train.Supervisor(logdir=checkpoint_dir, save_summaries_secs=0, saver=None)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with sv.managed_session(config=config) as sess:
        logging.info("Attempting to resume training from %s...", checkpoint_dir)
        checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        logging.info("Last checkpoint found: %s", checkpoint)
        if checkpoint:
            saver.restore(sess, checkpoint)
        elif imagenet_ckpt:
            logging.info("Restoring pretrained weights from %s", imagenet_ckpt)
            train_model.imagenet_init_restorer.restore(sess, imagenet_ckpt)

        logging.info("Training...")
        start_time = time.time()
        last_summary_time = time.time()
        steps_per_epoch = train_model.reader.steps_per_epoch
        step = 1
        while step <= train_steps:
            fetches = {
                "train": train_model.train_op,
                "global_step": train_model.global_step,
            }
            if step % summary_freq == 0:
                fetches["loss"] = train_model.total_loss
                fetches["summary"] = sv.summary_op

            if debug:
                fetches.update(train_model.exports)

            results = sess.run(fetches)
            global_step = results["global_step"]

            if step % summary_freq == 0:
                sv.summary_writer.add_summary(results["summary"], global_step)
                train_epoch = math.ceil(global_step / steps_per_epoch)
                train_step = global_step - (train_epoch - 1) * steps_per_epoch
                this_cycle = time.time() - last_summary_time
                last_summary_time += this_cycle
                logging.info(
                    "Epoch: [%2d] [%5d/%5d] time: %4.2fs (%ds total) loss: %.3f",
                    train_epoch,
                    train_step,
                    steps_per_epoch,
                    this_cycle,
                    time.time() - start_time,
                    results["loss"],
                )

            if debug:
                debug_dir = os.path.join(checkpoint_dir, "debug")
                if not gfile.Exists(debug_dir):
                    gfile.MkDir(debug_dir)
                for name, tensor in results.items():
                    if name == "summary":
                        continue
                    s = io.BytesIO()
                    filename = os.path.join(debug_dir, name)
                    np.save(s, tensor)
                    with gfile.Open(filename, "w") as f:
                        f.write(s.getvalue())
                # _print_losses(os.path.join(checkpoint_dir, 'debug'))
                return

            # steps_per_epoch == 0 is intended for debugging, when we run with a
            # single image for sanity check
            if steps_per_epoch == 0 or step % steps_per_epoch == 0:
                logging.info("[*] Saving checkpoint to %s...", checkpoint_dir)
                saver.save(
                    sess, os.path.join(checkpoint_dir, "model"), global_step=global_step
                )

            # Setting step to global_step allows for training for a total of
            # train_steps even if the program is restarted during training.
            step = global_step + 1


if __name__ == "__main__":
    main()
