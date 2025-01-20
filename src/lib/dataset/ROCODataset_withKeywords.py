import json
import os.path

from torchvision.datasets import VisionDataset
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import transforms
from typing import Callable, Optional
import unicodedata
from PIL import Image


class ImageTextDataset_withKeywords(VisionDataset):
    """
    Dtaset for loading image-text data for tasks like CLIP training, Image Captioning.

    Args:
        root: (string): The root path where the dataset is stored
        file_path: (string): Path to the file containing the image_paths and associated captions.
            The expected format is jsonlines where each line is a json object containing to keys.
            `image_path`: The path to the image.
            `captions`: An `array` of captions.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
            self,
            name,
            cfg,
            root="",
            max_seq_length=70,
            transform: Optional[Callable] = None
    ):
        super().__init__(root, transform)
        self.input_size = cfg["input_size"]
        self.cfg = cfg
        self.mode = name
        self.transform = transform
        self.update_transform()

        if self.mode == "train":
            file_path = cfg["dataset"]["train_json"]
        elif self.mode == "eval" or name == "val":
            file_path = cfg["dataset"]["validation_json"]
        elif self.mode == "test":
            file_path = cfg["dataset"]["test_json"]
        else:
            raise ValueError(f"{name} dataset is not supported!")

        file_path_kw = str(ImageTextDataset_withKeywords.remove_last_two_parts(file_path))
        if self.mode == "train":
            file_path_kw = os.path.join(file_path_kw,  "data", "train", "radiology")
        elif self.mode == "eval" or name == "val":
            file_path_kw = os.path.join(file_path_kw, "data", "validation", "radiology")
        elif self.mode == "test":
            file_path_kw = os.path.join(file_path_kw, "data", "test", "radiology")
        else:
            raise ValueError(f"{name} dataset is not supported!")

        with open(file_path, "r") as f:
            examples = [json.loads(line) for line in f.readlines()]

            keyword_file = [os.path.basename(example["image_path"][:-4])
                            for example in examples]

        self.keywords = []
        self.captions = []
        self.image_paths = []
        self.max_seq_length = cfg["train"]["max_seq_length"]

        for i, example in enumerate(examples):
            # self.captions.append(example["caption"])
            # self.image_paths.append(example["image_path"])
            # caption_words = example["caption"].strip().split(" ")
            # trimmed_caption_words = caption_words[:self.max_seq_length]
            # caption = " ".join(trimmed_caption_words)
            # self.captions.append(example["caption"][:self.max_seq_length])  # TODO: check if this is correct <- IT IS NOT
            self.captions.append(example["caption"])  # TODO: check if this is correct <- THIS IS!!!
            # self.captions.append(caption)
            self.image_paths.append(example["image_path"])

            with open(os.path.join(
                    file_path_kw,
                    "keywords.txt"),
                    "r") as f:
                tag = None
                while tag != keyword_file[i]:
                    text = f.readline()
                    if text is None:
                        break
                    spt = text.split("\t")
                    tag, kws = spt[0], spt[1:]

                self.keywords.append(
                    kws
                )

            # self.image_paths.extend([example["image_path"]] * captions_per_image)

        self.captions = [unicodedata.normalize("NFKD", c) for c in self.captions]

    @staticmethod
    def remove_last_two_parts(path):
        # Remove last two parts of the path
        pth = os.path.join(*os.path.split(os.path.dirname(path))[:-1])
        return pth

    def _load_image(self, idx: int):
        path = self.image_paths[idx]
        try:
            image = read_image(path, mode=ImageReadMode.RGB)
        except RuntimeError:
            print(self.image_paths[idx])
            import torch
            image = torch.zeros((3, 200, 200))
        # image = Image.open(path)
        return image

    def _load_target(self, idx):
        return self.captions[idx]

    def load_keywords(self, idx):
        return self.keywords[idx], self.captions[idx]

    def __getitem__(self, index: int):
        image = self._load_image(index)
        target = self._load_target(index)

        image = self.transform(image)

        return image, target

    def __len__(self) -> int:
        return len(self.captions)

    def update_transform(self):
        if self.mode == "train":
            pipeline = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(
                    size=self.input_size,
                    scale=(0.08, 1.0),
                    ratio=(0.75, 1.333333333),
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            pipeline = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(int(self.input_size[0] / 0.875)),
                transforms.CenterCrop(self.input_size[0]),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        self.transform = pipeline


if __name__ == "__main__":
    from src.lib.data_transform.collate import collate_clip

    cfg_pth = "/home/felipe/Projects/roco-image-captioning/config.yaml"
    import yaml
    from tqdm import tqdm

    with open(cfg_pth, "r") as f:
        cfg = yaml.safe_load(f)
    ds = ImageTextDataset("train", cfg)
    from torch.utils.data import DataLoader

    dl = DataLoader(ds, batch_size=4, shuffle=True,
                    collate_fn=collate_clip, )

    for x, y in tqdm(dl):
        pass

    # for x, y in dl:
    #     print(x)
    #     print(y)
    #     break
