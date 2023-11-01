import torch.optim as optim
import torch.nn as nn
import lightning as L
from dataloader import DataModule
from model import LitModel
from pytorch_lightning.loggers import WandbLogger
from argparse import ArgumentParser


# %%wandb
def main():
    wandb_logger = WandbLogger(log_model='all')
    dm = DataModule()
    model = LitModel(*dm.dims, dm.num_classes, hidden_size=256)
    trainer = L.Trainer(
        max_epochs=5,
        accelerator="auto",
        devices=1,
        logger=None
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    # parser = ArgumentParser()
    # parser.add_argument("--accelerator", default=None)
    # args = parser.parse_args()

    main()

