import torch
from torch.utils.data import default_collate
from transformers import CLIPProcessor

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def process_text_and_images(images, targets):
    inputs = processor(text=targets, images=images, return_tensors="pt", padding=True, do_rescale=False, truncate=True)
    return inputs

def collate_clip(batch):
    default_batch = default_collate(batch)
    images = default_batch[0]
    targets = default_batch[1]
    processed = process_text_and_images(images, targets)
    return processed
