import lightning as L
from dataloader import DataModule
from VITmodel import ViT
from pytorch_lightning.loggers import WandbLogger


def main(hparams=[]):
    dm = DataModule()
    wandb_logger = WandbLogger(project='vit-model', log_model='all')

    model = ViT(img_size=32, patch_size=16, in_chans=3, num_classes=10,
                embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                qkv_bias=False, drop_rate=0.1, )

    trainer = L.Trainer(
        max_epochs=10,
        accelerator="auto",
        devices=1,
        logger=wandb_logger)

    trainer.fit(model, dm)


if __name__ == "__main__":
    # parser = ArgumentParser()
    # parser.add_argument("--epochs", default=None)
    # args = parser.parse_args()

    main() #args)

