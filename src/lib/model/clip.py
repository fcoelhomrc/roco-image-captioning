import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from transformers import CLIPModel

import torchmetrics
import einops

import lightning as L


class MedCLIP(L.LightningModule):

    def __init__(self, user_parameters):
        super().__init__()

        self.user_parameters = user_parameters

        # TODO: parametrize (processor needs to match, but it is only defined on custom collate_fn)
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32",
                                               attn_implementation="sdpa")
        self.loss_fn = self.scaled_pairwise_cosine_sim_loss
        self.save_hyperparameters()  # wandb

    def forward(self, pairs):
        text_features = self.model.get_text_features(input_ids=pairs["input_ids"],
                                                     attention_mask=pairs["attention_mask"], )
        image_features = self.model.get_image_features(pixel_values=pairs["pixel_values"], )
        return text_features, image_features

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [
                {"params": self.model.parameters(),
                 "name": "clip"},
            ],
            lr=self.user_parameters['optimizer']['lr'],
            weight_decay=self.user_parameters['optimizer']['weight_decay'],
        )
        return optimizer

    def training_step(self, train_batch, batch_idx):
        pairs, labels = train_batch
        text_features, image_features = self.forward(pairs)

        text_loss, image_loss = MedCLIP._compute_text_and_image_loss_terms(text_features, image_features, labels)
        loss = self.loss_fn(text_loss, image_loss, weight=0.5)

        self.log("train/loss", loss)  # wandb
        self.log("train/text_loss", text_loss)  # wandb
        self.log("train/image_loss", image_loss)  # wandb
        return loss

    def validation_step(self, val_batch, batch_idx):
        pairs, labels = val_batch
        text_features, image_features = self.forward(pairs)

        text_loss, image_loss = MedCLIP._compute_text_and_image_loss_terms(text_features, image_features, labels)
        loss = self.loss_fn(text_loss, image_loss, weight=0.5)

        self.log("val/loss", loss)  # wandb
        self.log("val/text_loss", text_loss)  # wandb
        self.log("val/image_loss", image_loss)  # wandb
        return loss

    @staticmethod
    def scaled_pairwise_cosine_sim_loss(text_loss, image_loss, weight=0.5):
        return weight * text_loss + (1 - weight) * image_loss

    @staticmethod
    def _compute_text_and_image_loss_terms(text_features, image_features, labels):
        text_logits = torch.bmm(einops.rearrange(text_features, "b d -> b 1 d"),
                                einops.rearrange(image_features, "b d -> b d 1"))
        image_logits = torch.bmm(einops.rearrange(image_features, "b d -> b 1 d"),
                                 einops.rearrange(image_features, "b d -> b d 1"))
        text_logits = einops.rearrange(text_logits, "b 1 1 -> b")
        image_logits = einops.rearrange(image_logits, "b 1 1 -> b")
        normalization = torch.linalg.vector_norm(text_features, dim=1) * torch.linalg.norm(image_features, dim=1)
        text_logits /= normalization
        image_logits /= normalization
        text_logits = torch.log(text_logits)
        image_logits = torch.log(image_logits)
        text_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            text_logits, labels
        )
        image_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            image_logits, labels
        )
        return image_loss, text_loss
