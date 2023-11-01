import lightning as L
from dataloader import DataModule
from model import LitModel
from pytorch_lightning.loggers import WandbLogger
from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint


# %%wandb
def main(hparams):
    wandb_logger = WandbLogger(project='test-model', log_model='all')
    dm = DataModule()
    model = LitModel(*dm.dims, dm.num_classes, hidden_size=256)
    checkpoint_callback = ModelCheckpoint(dirpath='../../checkpoints/', filename='{epoch}-{val_loss:.2f}', save_top_k=2)
    trainer = L.Trainer(
        max_epochs=hparams.epochs,
        accelerator="auto",
        devices=1,
        logger=wandb_logger
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--epochs", default=None)
    args = parser.parse_args()

    main(args)

