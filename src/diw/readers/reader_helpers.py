""" Module with MOTS2020 data reader helper functions """
import numpy as np


def id2imgpath(seq_idx, img_idx, input_dir):
    """creates path from image id
    Args:
        seq_idx (str): sequence id
        img_idx (str): image id
        input_dir (Path): path to the folder with the images
    Returns:
        str: path to the image
    """
    img_name = img_idx + ".jpg"
    img_path = input_dir / seq_idx / "img1" / img_name
    return str(img_path)


def id2depthpath(seq_idx, depth_idx, input_dir, depth_folder):
    """creates path from depth image id
    Args:
        seq_idx (str): sequence id
        img_idx (str): segmentation image id
        input_dir (Path): path to the folder with the segmentation images
    Returns:
        str: path to the depth image
    """
    depth_name = depth_idx + ".npy"
    depth_path = input_dir / seq_idx / depth_folder / depth_name
    return str(depth_path)


def path2id(path):
    """extracts image name from absolute path
    Args:
        path (Path): absolute path to a file
    Returns:
        str: name of the file
    """
    str_path = str(path).split("/")[-1]
    return str_path.split(".")[0]


def read_file_names(path):
    """read all file names in a given path
    Args:
        path (Path): absolute path to a directory
    Returns:
        list(str): file names
    """
    return [str(file_name) for file_name in path.glob("**/*")]


def read_mot_bb_file(path):
    """Reads bounding box mot file
    Args:
        path (str): path to the ground truth file
    Returns:
        bb_data (ndarray): array in [frame_id, ped_id, x, y, w, h] format
    """
    bb_file = open(path, "r")
    bb_lines = bb_file.readlines()
    bb_data = np.zeros(
        (len(bb_lines), 6), dtype=np.uint64
    )  # all coordinates are integers
    for i, line in enumerate(bb_lines):
        frame_id, ped_id, x, y, w, h = line.split(",")[:6]
        bb_data[i, ...] = np.array([frame_id, ped_id, x, y, w, h], dtype=np.float64)
    bb_file.close()
    return bb_data


def read_mot_seg_file(path):
    """Reads segmentation mot file
    Args:
        path (str): path to the ground truth file
    Returns:
        seg_data (ndarray): array in [frame_id, ped_id, h, w] format
        mask_strings (list): list of raw coco masks strings
    """
    seg_file = open(path, "r")
    seg_lines = seg_file.readlines()
    seg_data = np.zeros((len(seg_lines), 4), dtype=np.uint64)
    mask_strings = [None] * len(seg_lines)
    for i, line in enumerate(seg_lines):
        frame_id, ped_id, _, height, width, mask_string = line.split(" ")
        seg_data[i, ...] = np.array([frame_id, ped_id, height, width], dtype=np.uint64)
        mask_strings[i] = mask_string.strip()
    seg_file.close()
    return (seg_data, mask_strings)
