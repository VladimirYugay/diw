""" Script for running depth inference with the model from the checkpoint """
import logging
import os
import sys
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import tensorflow
import tensorflow.compat.v1 as tf
from IPython.core import ultratb
from PIL import Image

import diw
from diw.model import Model

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


def get_vars_to_save_and_restore(ckpt=None):
    """Returns list of variables that should be saved/restored.
    Args:
      ckpt: Path to existing checkpoint.  If present, returns only the subset of
          variables that exist in given checkpoint.
    Returns:
      List of all variables that need to be saved/restored.
    """
    model_vars = tf.trainable_variables()
    # Add batchnorm variables.
    bn_vars = [
        v
        for v in tf.global_variables()
        if "moving_mean" in v.op.name
        or "moving_variance" in v.op.name
        or "mu" in v.op.name
        or "sigma" in v.op.name
        or "global_scale_var" in v.op.name
    ]
    model_vars.extend(bn_vars)
    model_vars = sorted(model_vars, key=lambda x: x.op.name)
    mapping = {}
    if ckpt is not None:
        ckpt_var = tensorflow.train.list_variables(ckpt)
        ckpt_var_names = [name for (name, unused_shape) in ckpt_var]
        ckpt_var_shapes = [shape for (unused_name, shape) in ckpt_var]
        not_loaded = list(ckpt_var_names)
        for v in model_vars:
            if v.op.name not in ckpt_var_names:
                # For backward compatibility, try additional matching.
                v_additional_name = v.op.name.replace("egomotion_prediction/", "")
                if v_additional_name in ckpt_var_names:
                    # Check if shapes match.
                    ind = ckpt_var_names.index(v_additional_name)
                    if ckpt_var_shapes[ind] == v.get_shape():
                        mapping[v_additional_name] = v
                        not_loaded.remove(v_additional_name)
                        continue
                    else:
                        logging.warn("Shape mismatch, will not restore %s.", v.op.name)
                logging.warn(
                    "Did not find var %s in checkpoint: %s",
                    v.op.name,
                    os.path.basename(ckpt),
                )
            else:
                # Check if shapes match.
                ind = ckpt_var_names.index(v.op.name)
                if ckpt_var_shapes[ind] == v.get_shape() and v.op.name in not_loaded:
                    mapping[v.op.name] = v
                    not_loaded.remove(v.op.name)
                else:
                    logging.warn("Shape mismatch, will not restore %s.", v.op.name)
        if not_loaded:
            logging.warn("The following variables in the checkpoint were not loaded:")
            for varname_not_loaded in not_loaded:
                logging.info("%s", varname_not_loaded)
    else:  # just get model vars.
        for v in model_vars:
            mapping[v.op.name] = v
    return mapping


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
            plt.figure()
            plt.imshow(depth, plt.cm.get_cmap("plasma").reversed())
            img_name = str(img_path).split("/")[-1]
            plt.savefig(output_dir + "/" + img_name.replace(".jpg", ".png"))


if __name__ == "__main__":
    main()
