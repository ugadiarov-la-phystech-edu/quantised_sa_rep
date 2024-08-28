import os
import random
import sys

sys.path.append("..")

import numpy as np
import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import seed_everything

from argparse import ArgumentParser

from datasets import CLEVR
from models import SlotAttentionAE

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

DEFAULT_SEED = 42
# ------------------------------------------------------------
# Parse args
# ------------------------------------------------------------
parser = ArgumentParser()

parser.add_argument('--max_epochs', type=int, default=100)
parser.add_argument('--num_slots', type=int, default=7)

# add PROGRAM level args
program_parser = parser.add_argument_group('program')

# logger parameters
program_parser.add_argument("--log_model", default=True)

# dataset parameters
program_parser.add_argument("--train_path", type=str)

# Experiment parameters
program_parser.add_argument("--batch_size", type=int, default=64)
program_parser.add_argument("--from_checkpoint", type=str, default='')
program_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
program_parser.add_argument("--nums", type=int, nargs='+')
program_parser.add_argument("--project", type=str, required=True)
program_parser.add_argument("--group", type=str, required=True)
program_parser.add_argument("--run_name", type=str, required=True)
program_parser.add_argument("--sa_state_dict", type=str, default='./clevr7_od')
program_parser.add_argument("--pretrained", type=bool, default=False)
program_parser.add_argument("--beta", type=float, default=2.)
program_parser.add_argument("--num_workers", type=int, default=4)

# Add model specific args
# parser = SlotAttentionAE.add_model_specific_args(parent_parser=parser)

# Add all the available trainer options to argparse#
#parser = pl.Trainer.add_argparse_args(parser)

# Parse input
args = parser.parse_args()

# ------------------------------------------------------------
# Random
# ------------------------------------------------------------

seed_everything(args.seed, workers=True)

# ------------------------------------------------------------
# Logger
# ------------------------------------------------------------


# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------


train_dataset = CLEVR(images_path=os.path.join(args.train_path, 'images', 'train'),
                      scenes_path=os.path.join(args.train_path, 'scenes', 'CLEVR_train_scenes.json'),
                      max_objs=6)

val_dataset = CLEVR(images_path=os.path.join(args.train_path, 'images', 'val'),
                    scenes_path=os.path.join(args.train_path, 'scenes', 'CLEVR_val_scenes.json'),
                    max_objs=6)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True,
                          drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False,
                        drop_last=True)

# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

# model
dict_args = vars(args)

autoencoder = SlotAttentionAE(**dict_args)

project_name = 'object_discovery_CLEVR'

wandb_logger = WandbLogger(project=args.project, group=args.group, name=args.run_name, log_model=True)
# ------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------


monitor = 'Validation MSE'

# checkpoints
save_top_k = 1
checkpoint_callback = ModelCheckpoint(monitor=monitor, save_top_k=save_top_k)
every_epoch_callback = ModelCheckpoint(every_n_epochs=10, monitor=monitor)
# Learning rate monitor
lr_monitor = LearningRateMonitor(logging_interval='step')

# logger_callback = SlotAttentionLogger(val_samples=next(iter(val_loader)))

callbacks = [
    checkpoint_callback,
    # logger_callback,
    every_epoch_callback,
    # swa,
    # early_stop_callback,
    lr_monitor,
]

# if args.pretrained:
state_dict = torch.load(args.sa_state_dict)
autoencoder.load_state_dict(state_dict=state_dict, strict=False)

# ------------------------------------------------------------
# Trainer
# ------------------------------------------------------------
# trainer parameters
profiler = None  # 'simple'/'advanced'/None
accelerator = 'gpu'
devices = [int(args.devices)]

# trainer
trainer = pl.Trainer(accelerator='gpu',
                     devices=[0],
                     max_epochs=args.max_epochs,
                     profiler=profiler,
                     callbacks=callbacks,
                     logger=wandb_logger,
                     )
#  precision=16,
# deterministic=False)

if not len(args.from_checkpoint):
    args.from_checkpoint = None

# Train
trainer.fit(autoencoder, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=args.from_checkpoint)
# Test
trainer.test(dataloaders=val_loader, ckpt_path=None)
