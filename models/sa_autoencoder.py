import os

import numpy as np
import pytorch_lightning as pl
import torch
import wandb

from torch import nn
from torch.nn import functional as F
from torch.optim import lr_scheduler

from modules import Decoder, PosEmbeds, CoordQuantizer
from modules.slot_attention import SlotAttentionBase
from utils import spatial_broadcast, spatial_flatten


def visualize(images):
    B, _, H, W = images[0].shape  # first image is observation
    viz_imgs = []
    for _img in images:
        if len(_img.shape) == 4:
            viz_imgs.append(_img)
        else:
            viz_imgs += list(torch.unbind(_img, dim=1))
    viz_imgs = torch.cat(viz_imgs, dim=-1)
    # return torch.cat(torch.unbind(viz_imgs,dim=0), dim=-2).unsqueeze(0)
    return viz_imgs


def for_viz(x):
    return np.array(
        (x.clamp(-1, 1).permute(0, 2, 3, 1).detach().cpu().numpy() + 1) / 2 * 255.0, dtype=np.uint8
    )


class SlotAttentionAE(pl.LightningModule):
    """
    Slot attention based autoencoder for object discovery task
    """

    def __init__(self,
                 resolution=(128, 128),
                 num_slots=7,
                 num_iters=3,
                 in_channels=3,
                 slot_size=64,
                 hidden_size=64,
                 beta=2,
                 lr=4e-4,
                 num_steps=int(3e5), **kwargs
                 ):
        super().__init__()
        self.resolution = resolution
        self.num_slots = num_slots
        self.num_iters = num_iters
        self.in_channels = in_channels
        self.slot_size = slot_size
        self.hidden_size = hidden_size
        self.log_images = 8

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_size, kernel_size=5, padding=(2, 2)), nn.ReLU(),
            *[nn.Sequential(nn.Conv2d(hidden_size, hidden_size, kernel_size=5, padding=(2, 2)), nn.ReLU()) for _ in
              range(3)]
        )
        self.decoder_initial_size = (8, 8)

        # Decoder
        self.decoder = Decoder()

        self.enc_emb = PosEmbeds(64, self.resolution)
        self.dec_emb = PosEmbeds(64, self.decoder_initial_size)

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, slot_size)
        )
        self.slots_lin = nn.Linear(hidden_size * 2, hidden_size)

        self.slot_attention = SlotAttentionBase(num_slots=num_slots, iters=num_iters, dim=slot_size,
                                                hidden_dim=slot_size * 2)
        self.coord_quantizer = CoordQuantizer()
        self.automatic_optimization = False
        self.num_steps = num_steps
        self.lr = lr
        self.beta = beta
        self.save_hyperparameters()

    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.enc_emb(x)

        x = spatial_flatten(x[0])
        x = self.layer_norm(x)
        x = self.mlp(x)

        slots = self.slot_attention(x)

        props, coords, kl_loss = self.coord_quantizer(slots)
        slots = torch.cat([props, coords], dim=-1)
        slots = self.slots_lin(slots)

        x = spatial_broadcast(slots, self.decoder_initial_size)
        x = self.dec_emb(x)
        x = self.decoder(x[0])

        x = x.reshape(inputs.shape[0], self.num_slots, *x.shape[1:])
        recons, masks = torch.split(x, self.in_channels, dim=2)
        masks = F.softmax(masks, dim=1)
        recons = recons * masks
        result = torch.sum(recons, dim=1)
        return result, recons, kl_loss

    def step(self, batch, return_result=False):
        imgs = batch['image']
        result, recons, kl_loss = self(imgs)
        loss = F.mse_loss(result, imgs)
        if return_result:
            return loss, kl_loss, result, recons

        return loss, kl_loss

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        sch = self.lr_schedulers()
        optimizer = optimizer.optimizer

        loss, kl_loss = self.step(batch)
        self.log('Training MSE', loss, on_step=False, on_epoch=True)
        self.log('Training KL', kl_loss, on_step=False, on_epoch=True)

        loss = loss + kl_loss * self.beta
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sch.step()

        self.log('lr', sch.get_last_lr()[0], on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, kl_loss, result, recons = self.step(batch, return_result=True)
        self.log('Validation MSE', loss, on_step=False, on_epoch=True)
        self.log('Validation KL', kl_loss, on_step=False, on_epoch=True)

        if batch_idx == 0:
            imgs = batch['image'][:self.log_images]
            result = result[:self.log_images]
            recons = recons[:self.log_images]
            self.logger.log_image(key="samples", images=list(for_viz(visualize([imgs, result, recons]))))

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr, total_steps=self.num_steps, pct_start=0.05)
        return [optimizer], [scheduler]
