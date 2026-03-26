import pytorch_lightning as pl
from omegaconf import OmegaConf

from src.lightning.module import PINNModule
from src.lightning.datamodule import PhaseDataModule

def main():
    cfg = OmegaConf.load("configs/config.yaml")

    datamodule = PhaseDataModule(cfg)
    model = PINNModule(cfg)

    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator="gpu" if cfg.device == "gpu" else "cpu",
        devices=1
    )

    trainer.fit(model, datamodule)

if __name__ == "__main__":
    main()