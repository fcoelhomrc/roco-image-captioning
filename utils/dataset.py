import os
import numpy as np
import matplotlib.pyplot as plt

from dataloading_tools import compute_captions_index, read_caption, read_image

DATA_DIR = os.path.join("..", "roco-dataset", "data")
SPLITS = ["train", "validation", "test"]

for SPLIT in SPLITS:
    IMAGES_RADIOLOGY = os.path.join(DATA_DIR, SPLIT, "radiology", "images")
    IMAGES_NON_RADIOLOGY = os.path.join(DATA_DIR, SPLIT, "non-radiology", "images")

    CAPTIONS_RADIOLOGY = os.path.join(DATA_DIR, SPLIT, "radiology", "captions.txt")
    CAPTIONS_NON_RADIOLOGY = os.path.join(DATA_DIR, SPLIT, "non-radiology", "captions.txt")

    INDEX_RADIOLOGY = compute_captions_index(CAPTIONS_RADIOLOGY)
    INDEX_NON_RADIOLOGY = compute_captions_index(CAPTIONS_NON_RADIOLOGY)

    rng = np.random.default_rng(42)
    ### rad ###
    random_sample = rng.choice(list(INDEX_RADIOLOGY.keys()))
    random_caption = read_caption(CAPTIONS_RADIOLOGY,
                                  random_sample, INDEX_RADIOLOGY)
    print(random_sample, random_caption)

    random_image = read_image(IMAGES_RADIOLOGY, random_sample)
    if random_image:
        plt.imshow(random_image)
    plt.show()

    ### non rad ###
    random_sample = rng.choice(list(INDEX_NON_RADIOLOGY.keys()))
    random_caption = read_caption(CAPTIONS_NON_RADIOLOGY,
                                  random_sample, INDEX_NON_RADIOLOGY)
    print(random_sample, random_caption)

    random_image = read_image(IMAGES_NON_RADIOLOGY, random_sample)
    if random_image:
        plt.imshow(random_image)

    plt.show()