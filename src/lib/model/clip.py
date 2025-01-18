import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel

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
        self.register_buffer("labels", torch.arange(self.user_parameters["train"]["batch_size"]))
        self.save_hyperparameters()  # wandb

    def forward(self, inputs):
        # compute similarities (contrastive pairs)
        outputs = self.model(**inputs)
        logits_per_text = outputs.logits_per_text
        logits_per_image = outputs.logits_per_image
        return logits_per_text, logits_per_image

    def extract_features(self, inputs):
        text_features = self.model.get_text_features(input_ids=inputs["input_ids"],
                                                     attention_mask=inputs["attention_mask"], )
        image_features = self.model.get_image_features(pixel_values=inputs["pixel_values"], )
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

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=self.user_parameters["optimizer"]["lr_scheduler_rate"],
            patience=self.user_parameters["optimizer"]["lr_scheduler_patience"],
        )

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": "train/loss"}

    def training_step(self, train_batch, batch_idx):
        # compute similarities (contrastive pairs)
        outputs = self.model(**train_batch)
        logits_per_text = outputs.logits_per_text
        logits_per_image = outputs.logits_per_image
        # compute loss
        text_loss, image_loss = MedCLIP.compute_text_and_image_loss_terms(
            logits_per_text, logits_per_image, self.labels
        )
        loss = self.loss_fn(text_loss, image_loss, weight=0.5)

        # metrics
        probas = logits_per_image.softmax(dim=1)
        preds = probas.argmax(dim=1)
        accuracy = torchmetrics.functional.accuracy(
            preds, self.labels, task="multiclass", num_classes=len(self.labels)
        )
        mcc = torchmetrics.functional.matthews_corrcoef(
            preds, self.labels, task="multiclass", num_classes=len(self.labels)
        )
        self.log("train/loss", loss)  # wandb
        self.log("train/text_loss", text_loss)  # wandb
        self.log("train/image_loss", image_loss)  # wandb
        self.log("train/accuracy", accuracy)  # wandb
        self.log("train/mcc", mcc.to(torch.float32))  # wandb
        return loss

    def validation_step(self, val_batch, batch_idx):
        # compute similarities (contrastive pairs)
        outputs = self.model(**val_batch)
        logits_per_text = outputs.logits_per_text
        logits_per_image = outputs.logits_per_image
        # compute loss
        text_loss, image_loss = MedCLIP.compute_text_and_image_loss_terms(
            logits_per_text, logits_per_image, self.labels
        )
        loss = self.loss_fn(text_loss, image_loss, weight=0.5)

        # metrics
        probas = logits_per_image.softmax(dim=1)
        preds = probas.argmax(dim=1)
        accuracy = torchmetrics.functional.accuracy(
            preds, self.labels, task="multiclass", num_classes=len(self.labels)
        )
        mcc = torchmetrics.functional.matthews_corrcoef(
            preds, self.labels, task="multiclass", num_classes=len(self.labels)
        )
        self.log("val/loss", loss)  # wandb
        self.log("val/text_loss", text_loss)  # wandb
        self.log("val/image_loss", image_loss)  # wandb
        self.log("val/accuracy", accuracy)  # wandb
        self.log("val/mcc", mcc.to(torch.float32))  # wandb
        return loss

    def test_step(self, test_batch, batch_idx):
        # compute similarities (contrastive pairs)
        outputs = self.model(**test_batch)
        logits_per_text = outputs.logits_per_text
        logits_per_image = outputs.logits_per_image
        # compute loss
        text_loss, image_loss = MedCLIP.compute_text_and_image_loss_terms(
            logits_per_text, logits_per_image, self.labels
        )
        loss = self.loss_fn(text_loss, image_loss, weight=0.5)

        # metrics
        probas = logits_per_image.softmax(dim=1)
        preds = probas.argmax(dim=1)
        accuracy = torchmetrics.functional.accuracy(
            preds, self.labels, task="multiclass", num_classes=len(self.labels)
        )
        mcc = torchmetrics.functional.matthews_corrcoef(
            preds, self.labels, task="multiclass", num_classes=len(self.labels)
        )
        self.log("test/loss", loss)  # wandb
        self.log("test/text_loss", text_loss)  # wandb
        self.log("test/image_loss", image_loss)  # wandb
        self.log("test/accuracy", accuracy)  # wandb
        self.log("test/mcc", mcc.to(torch.float32))  # wandb
        return loss


    @staticmethod
    def scaled_pairwise_cosine_sim_loss(text_loss, image_loss, weight=0.5):
        return weight * text_loss + (1 - weight) * image_loss

    @staticmethod
    def compute_text_and_image_loss_terms(logits_per_text, logits_per_image, labels):
        text_loss = torch.nn.functional.cross_entropy(
            logits_per_text, labels
        )
        image_loss = torch.nn.functional.cross_entropy(
            logits_per_image, labels
        )
        return image_loss, text_loss
