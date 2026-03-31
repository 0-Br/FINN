import os
import glob
import argparse
import shutil
import json
from omegaconf import OmegaConf

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from data.dataset import load_dataset
from models.UNet import UNetConfig, UNet
from models.PINN import PINNConfig, PINN

from utils.learn import get_cosine_schedule_with_warmup
from utils.metrics import Critic


torch.set_float32_matmul_precision("medium")
pl.seed_everything(3407)


class FINN(pl.LightningModule):
    """Prediction"""

    def __init__(self, config, model_config):
        """"""
        super().__init__()
        self.config = OmegaConf.load(config) if isinstance(config, str) else config
        if isinstance(model_config, str):
            with open(model_config) as f:
                self.model_config = json.load(f)
        else:
            self.model_config = model_config

        if self.model_config["model_type"] == "UNet":
            self.model: nn.Module = UNet(UNetConfig(**self.model_config))
        elif self.model_config["model_type"] == "PINN":
            self.model: nn.Module = PINN(PINNConfig(**self.model_config))
        else:
            raise TypeError("There's No such Model!")

        print(f"Model: {self.model_config['name']}")

        self.train_dataset, self.valid_dataset, self.test_dataset = load_dataset(config.data_dir)
        print(f"Volume of train data: {len(self.train_dataset)}")
        print(f"Volume of validation data: {len(self.valid_dataset)}")
        print(f"Volume of test data: {len(self.test_dataset)}")

        self.critic_train = Critic()
        self.critic_valid = Critic()
        self.critic_test = Critic()

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        x = batch["x"]
        r = batch["r"]
        y = batch["y"]
        outputs = self.forward(x, r, y)
        self.log("train_loss", outputs.loss)
        if self.trainer.is_global_zero:
            self.critic_train.record(y.reshape(-1).detach().cpu().numpy(), outputs.logits.reshape(-1).detach().cpu().numpy())
        return outputs.loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x = batch["x"]
        r = batch["r"]
        y = batch["y"]
        outputs = self.forward(x, r, y)
        self.log("val_loss", outputs.loss)
        self.critic_valid.record(y.reshape(-1).detach().cpu().numpy(), outputs.logits.reshape(-1).detach().cpu().numpy())
        return outputs.loss

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        x = batch["x"]
        r = batch["r"]
        y = batch["y"]
        outputs = self.forward(x, r, y)
        test_loss = outputs.loss
        self.log("test_loss", test_loss)
        # For test data, record metrics for evaluation
        self.critic_test.record(y.reshape(-1).detach().cpu().numpy(), outputs.logits.reshape(-1).detach().cpu().numpy())
        return test_loss

    @torch.no_grad()
    def on_train_epoch_end(self):
        self.critic_train.judge()
        self.log("train_r2",        self.critic_train.r2)
        self.log("train_mse",       self.critic_train.mse)
        self.log("train_mae",       self.critic_train.mae)
        self.critic_train.clear()

    @torch.no_grad()
    def on_validation_epoch_end(self):
        self.critic_valid.judge()
        self.log("val_r2",        self.critic_valid.r2)
        self.log("val_mse",       self.critic_valid.mse)
        self.log("val_mae",       self.critic_valid.mae)
        self.critic_valid.clear()

    @torch.no_grad()
    def on_test_epoch_end(self):
        self.critic_test.judge()
        self.log("test_r2", self.critic_test.r2)
        self.log("test_mse", self.critic_test.mse)
        self.log("test_mae", self.critic_test.mae)
        self.critic_test.clear()

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), self.config.lr) # optimizer
        warmup_step = int(self.config.num_warming_steps)
        print("Warmup step: ", warmup_step)
        # warmup_strategy: note the unit: step, need to adjust
        schedule = {
            "scheduler": get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warmup_step, num_training_steps=self.config.num_training_steps),
            "interval": "step",
            "frequency": 1}
        return [optimizer], [schedule]

    def configure_callbacks(self):
        lr_monitor = LearningRateMonitor(logging_interval="step")
        checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(self.config.save_dir, "checkpoints"), filename="{epoch}-{val_loss:.4f}", monitor="val_loss", mode="min", save_last=True, save_top_k=10, save_weights_only=False)
        earlystop_callback = EarlyStopping(monitor="val_loss", patience=self.config.es_patience, mode="min")
        return [lr_monitor, checkpoint_callback, earlystop_callback]

    def train_dataloader(self):
        dataset = self.train_dataset
        train_loader = DataLoader(dataset, batch_size=self.config.batch_size, num_workers=self.config.num_workers, shuffle=True)
        return train_loader

    def val_dataloader(self):
        dataset = self.valid_dataset
        valid_loader = DataLoader(dataset, batch_size=self.config.batch_size, num_workers=self.config.num_workers, shuffle=False)
        return valid_loader

    def test_dataloader(self):
        dataset = self.test_dataset
        test_loader = DataLoader(dataset, batch_size=self.config.batch_size, num_workers=self.config.num_workers, shuffle=False)
        return test_loader


def main(config, model_config):

    model = FINN(config, model_config)

    print("Model Architecture:")
    print(model)

    log_dir = "/".join(config.save_dir.split("/")[:-1])
    logger = TensorBoardLogger(save_dir=log_dir, name=config.save_dir.split("/")[-1])

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=config["devices"],
        logger=logger,
        log_every_n_steps=1, # steps to load tensorboard
        precision=config["precision"],
        max_steps=config["num_training_steps"],
        default_root_dir="./results")

    if config["pretrained_model"]:
        trainer.fit(model, ckpt_path=config["pretrained_model"])
    else:
        trainer.fit(model)


if __name__ == "__main__":
    device_id = torch.cuda.current_device()

    parser = argparse.ArgumentParser(description="Flood Prediction")
    parser.add_argument("--config", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--version", type=str)
    args = parser.parse_args()

    print("Args in experiment:")
    print(args)

    config = OmegaConf.load(args.config)

    with open(f"./models/config/{args.model}/{args.version}.json") as f:
        model_config = json.load(f)

    # save to path
    if device_id == config.devices[0]:
        os.makedirs(os.path.dirname(f"{config.save_dir}/config.yaml"), exist_ok=True)
        os.makedirs(os.path.dirname(f"{config.save_dir}/model.json"), exist_ok=True)
        shutil.copy(args.config, f"{config.save_dir}/config.yaml")
        shutil.copy(f"./models/config/{args.model}/{args.version}.json", f"{config.save_dir}/model.json")

    print("Save results to:", config.save_dir)

    main(config, model_config)
