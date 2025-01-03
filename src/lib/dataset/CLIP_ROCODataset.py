from ROCODataset import ImageTextDataset

import torch
from transformers import CLIPProcessor
from typing import Optional, Callable

class CLIPDataset(ImageTextDataset):

    def __init__(
        self,
        name,
        cfg,
        root="",
        max_seq_length=70,
        transform: Optional[Callable] = None,
    ):
        super().__init__(name, cfg, root, max_seq_length, transform)

        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


    def process_text_and_images(self, images, targets):
        inputs = self.processor(text=targets, images=images, return_tensors="pt", padding=True)
        return inputs


    def create_pairs(self, images, targets):
        size = images.shape[0]
        processed = self.process_text_and_images(images, targets)
        pairs = {"input_ids": [], "attention_mask": [], "pixel_values": []}
        labels = []
        for i in range(size):
            for j in range(size):

                # build every possible image / text pair
                pairs["input_ids"].append(processed["input_ids"][i])
                pairs["attention_mask"].append(processed["attention_mask"][i])  # masks padding added to text
                pairs["pixel_values"].append(processed["pixel_values"][j])

                # labels for contrastive learning
                if i == j:  # positive sample
                    labels.append(1)
                else:  # negative sample
                    labels.append(0)

        # convert to tensor
        for k, v in pairs.items():
            pairs[k] = torch.stack(v, dim=0)
        labels = torch.tensor(labels, dtype=torch.float32)
        return pairs, labels

    def __getitem__(self, index: int):
        image, target = super().__getitem__(index)
        pairs, labels = self.create_pairs(image, target)
        return pairs, labels

