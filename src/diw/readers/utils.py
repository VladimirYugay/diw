""" module with util functions """
import numpy as np
from PIL import Image
from pycocotools import mask as rletools


def resize_img(img, img_shape):
    """ resizes an image """
    return img.resize(img_shape, Image.LANCZOS).convert("RGB")


def resize_masks(masks, img_shape):
    """ resizes segmentation masks """
    resized_masks = np.zeros((masks.shape[0], img_shape[1], img_shape[0]))
    for i in range(masks.shape[0]):
        mask_img = Image.fromarray(np.uint8(masks[i] * 255))
        mask_img = mask_img.resize(img_shape, Image.NEAREST)
        mask_img = np.array(mask_img)
        mask_img[mask_img != 0] = 1
        resized_masks[
            i,
        ] = mask_img
    return resized_masks


def resize_boxes(boxes, old_shape, new_shape):
    """ resizes bounding boxes based on old and new shapes of the image """
    boxes = boxes.copy()
    boxes = np.asarray(boxes, dtype=np.float64)
    boxes[:, [0, 2]] *= new_shape[0] / old_shape[0]
    boxes[:, [1, 3]] *= new_shape[1] / old_shape[1]
    # sometimes boxes are outside of the image, we need to clip them
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, new_shape[0] - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, new_shape[1] - 1)
    return np.asarray(boxes, dtype=np.uint64)


def scale_intrinsics(intrinsics, old_shape, new_shape):
    """Scales intrinsics based on the image size change
    Args:
        intrinsics (ndarray): intrinsics matrix
        old_shape (tuple): old image dimensions  (hxw)
        new_shape (tuple): new image dimensions  (hxw)
    Returns:
        intrinsics (ndarray): rescaled intrinsics
    """
    intrinsics = intrinsics.copy()
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    intrinsics[0, :] *= new_shape[0] / old_shape[0]
    intrinsics[1, :] *= new_shape[1] / old_shape[1]
    return intrinsics


def patch_boxes(image, boxes):
    """Creates image patches with only box regions visible
    Args:
        image (ndarray): depth of RGB image of (h, w, c) or (h, w) shape
        boxes (ndarray): boxes to apply of (n, top_left, bottom_right) format
    Returns:
        patches (ndarray): patches of (n, h, w, c) or (n, h, w) shape
    """
    boxes = boxes.astype(np.int16)
    patches = np.full([boxes.shape[0], *image.shape], 0)
    for i, box in enumerate(boxes):
        width, height = box[2] - box[0], box[3] - box[1]
        patches[
            i,
            box[1] : box[1] + height,
            box[0] : box[0] + width,
        ] = image[box[1] : box[1] + height, box[0] : box[0] + width]
    return patches


def patch_masks(image, masks):
    """Creates image patches with only mask regions visible
    Args:
        image (ndarray): depth of RGB image of (h, w, c) or (h, w) shape
        masks (ndarray): masks to apply of (h, w) shape
    Returns:
        patches (ndarray): patches of (n, h, w, c) or (n, h, w) shape
    """
    patches = np.repeat(
        image[
            None,
        ],
        masks.shape[0],
        axis=0,
    )
    for i, mask in enumerate(masks):
        patches[i, :][mask == 0] = 0
    return patches


def decode_mask(height, width, mask_string):
    """ decodes coco segmentation mask string to numpy """
    return rletools.decode(
        {
            "size": [int(height), int(width)],
            "counts": mask_string.encode(encoding="UTF-8"),
        }
    )


def compute_box_center(box):
    """ compute center of the box represented with top left and bottom right corners"""
    return box[0] + (box[2] - box[0]) // 2, box[1] + (box[3] - box[1]) // 2


def compute_mask_center(mask):
    """ compute center of the binary mask """
    y, x = np.nonzero(mask)
    c_x, c_y = x.min() + (x.max() - x.min()) // 2, y.min() + (y.max() - y.min()) // 2
    return c_x, c_y


def mask_out(img, masks):
    """ masks out objects inside the image or depth """
    img = img.copy()
    combined_mask = masks.sum(axis=0)
    img[combined_mask == 1] = 0
    return img


def load_image(img_file):
    """Load image from disk. Output value range: [0,1]."""
    return Image.open(img_file).convert("RGB")


def save_img(img, path):
    """saves image array
    Args:
        img (ndarray): img array of h x w x c
        path (str): path to save image to
    """
    if 0 <= img.mean() <= 1.0:
        img = (img * 255).astype(np.uint8)
    img_type = "RGB" if len(img.shape) == 3 else "L"
    img = Image.fromarray(img, mode=img_type)
    img.save(path)
