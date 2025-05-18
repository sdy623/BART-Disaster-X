from bisect import insort
import os
from datetime import datetime
from time import perf_counter
from typing import Optional, Union

import torch

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, ProgressBar, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger, Logger
from lightning.pytorch.profilers import PyTorchProfiler

from datamodule import AutoTokenizerDataModule
from module import LinearBAModel
from utils import create_dirs, log_perf
from config import Config, DataModuleConfig, ModuleConfig, TrainerConfig

import ray.train.lightning
from pytorch_lightning.loggers import CometLogger
from dvclive.lightning import DVCLiveLogger

from clearml import Task

import optuna
from optuna.integration import PyTorchLightningPruningCallback
import argparse

# constants
model_name = ModuleConfig.model_name
dataset_name = DataModuleConfig.dataset_name

# paths
cache_dir = Config.cache_dir
log_dir = Config.log_dir
ckpt_dir = Config.ckpt_dir
prof_dir = Config.prof_dir
perf_dir = Config.perf_dir
# creates dirs to avoid failure if empty dir has been deleted
create_dirs([cache_dir, log_dir, ckpt_dir, prof_dir, perf_dir])

study_name = "bart-pytorch-finetune-linear"  # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format(study_name)
# set matmul precision
# see https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
torch.set_float32_matmul_precision("high")

MAXEPOCH = 8
BATCHSIZE = 8
def objective(trial: optuna.trial.Trial) -> float:
    """Configure the Optuna optimization objectives and the Lightning Trainer.

    Note:
        for all Trainer flags, see https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-flags
    """
    #Task.init(project_name="examples", task_name="BERT Disaster Tweets on Pytlightning")
    # ## LightningDataModule ## #
    lr = trial.suggest_float("lr_bert", 1e-5, 8e-5, log=True)
    # epoch = trial.suggest_int("eporch", 2, 8)
    # lr_lstm = trial.suggest_float("lr_lstm", 1e-4, 3e-3, log=True)
    # attn_dropout = trial.suggest_float("attn_dropout", 0.00, 0.1)
    # classifier_dropout = trial.suggest_float("classifier_dropout", 0.00, 0.05)
    # lr_gamma = trial.suggest_float("classifier_dropout", 0.6, 0.9)

    lit_datamodule = AutoTokenizerDataModule(
        model_name=model_name,
        dataset_name=dataset_name,
        cache_dir=cache_dir,
        batch_size=BATCHSIZE,
    )

    # ## LightningModule ## #
    lit_module = LinearBAModel(learning_rate=lr, learning_rate_bert=lr, attention_probs_dropout=0.1, 
                             classifier_dropout=0, lr_gamma=0.75)

    # ## Lightning Trainer callbacks, loggers, plugins ## #

    # set logger
    csvlogger = CSVLogger(
        save_dir=log_dir,
        name="csv-logs",
    )
    # logger=Logger()

    # set callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            monitor="val_f1",
            filename="model",
            save_top_k=3,
            mode="max",
            save_weights_only=True,
        ),
        LearningRateMonitor(logging_interval='step'),
        PyTorchLightningPruningCallback(trial, monitor="val_f1")

        ]

    # set profiler


    # ## create Trainer and call .fit ## #
    lit_trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        strategy="auto",
        precision=TrainerConfig.precision,
        max_epochs=MAXEPOCH,
        logger=[csvlogger, CometLogger(api_key="YOUR_COMET_API_KEY"), DVCLiveLogger(save_dvc_exp=True)],
        # logger=[csvlogger],
        callbacks=callbacks,
        log_every_n_steps=50,
        # strategy=ray.train.lightning.RayDDPStrategy(find_unused_parameters=True),
        # plugins=[ray.train.lightning.RayLightningEnvironment()],
        # callbacks=[ray.train.lightning.RayTrainReportCallback(), LearningRateMonitor(logging_interval='epoch')],
        # [1a] Optionally, disable the default checkpointing behavior
        # in favor of the `RayTrainReportCallback` above.
        # enable_checkpointing=False,
    )
    hyperparameters = dict(lr=lr)
    lit_trainer.logger.log_hyperparams(hyperparameters)
    lit_trainer.fit(model=lit_module, datamodule=lit_datamodule)
    return lit_trainer.callback_metrics["val_f1"].item()

if __name__ == "__main__":
    #from jsonargparse import CLI
    #CLI(train, as_positional=False)
    parser = argparse.ArgumentParser(description="PyTorch Lightning example.")
    parser.add_argument(
        "--pruning",
        "-p",
        action="store_true",
        help="Activate the pruning feature. `MedianPruner` stops unpromising "
        "trials at the early stages of training.",
    )
    args = parser.parse_args()

    pruner = optuna.pruners.PatientPruner(optuna.pruners.MedianPruner(), patience=3) if args.pruning else optuna.pruners.NopPruner()

    study = optuna.create_study(direction="maximize", pruner=pruner, study_name=study_name, storage=storage_name, load_if_exists=True)
    study.optimize(objective, n_trials=150, timeout=86400) # seconds

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))