import os
from PIL import Image

def compute_captions_index(file_path):
    index = {}
    with open(file_path, 'r') as f:
        while True:
            byte_position = f.tell()
            line = f.readline()
            if not line:
                break
            id_value = line.strip().split()
            if len(id_value) == 2:
                id_, value = id_value
                index[id_] = byte_position
    return index


def read_caption(file_path, id_, index):
    with open(file_path, 'r') as f:
        byte_position = index.get(id_)
        if byte_position is None:
            return None
        f.seek(byte_position)
        line = f.readline()
        _, value = line.strip().split()
        return value


def read_image(image_dir, idx):
    file_name = f"{idx}.jpg"
    file_path = os.path.join(image_dir, file_name)
    if os.path.exists(file_path):
        img = Image.open(file_path)
        return img
    return None