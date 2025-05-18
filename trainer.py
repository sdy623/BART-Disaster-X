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
from module import CustomModel
from utils import create_dirs, log_perf
from config import Config, DataModuleConfig, ModuleConfig, TrainerConfig

import ray.train.lightning
from pytorch_lightning.loggers import CometLogger
from dvclive.lightning import DVCLiveLogger

#from ray.train.torch.config import TorchConfig
#from ray.train.torch import TorchTrainer
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

study_name = "bert-pytorch-finetune-neo3"  # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format(study_name)
# set matmul precision
# see https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
torch.set_float32_matmul_precision("high")

MAXEPOCH = 8
BATCHSIZE = 8
def objective(trial: optuna.trial.Trial) -> float:
    """a custom Lightning Trainer utility

    Note:
        for all Trainer flags, see https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-flags
    """
    #Task.init(project_name="examples", task_name="BERT Disaster Tweets on Pytlightning")
    # ## LightningDataModule ## #
    lr = trial.suggest_float("lr_bert", 1e-5, 8e-5, log=True)
    #epoch = trial.suggest_int("eporch", 2, 8)
    #lr_lstm = trial.suggest_float("lr_lstm", 1e-4, 3e-3, log=True)
    #attn_dropout = trial.suggest_float("attn_dropout", 0.00, 0.1)
    #classifier_dropout = trial.suggest_float("classifier_dropout", 0.00, 0.05)
    # lr_gamma = trial.suggest_float("classifier_dropout", 0.6, 0.9)

    lit_datamodule = AutoTokenizerDataModule(
        model_name=model_name,
        dataset_name=dataset_name,
        cache_dir=cache_dir,
        batch_size=BATCHSIZE,
    )

    # ## LightningModule ## #
    lit_module = CustomModel(learning_rate=lr, learning_rate_bert=lr, attention_probs_dropout=0.1, 
                             classifier_dropout=0, lr_gamma=0.75)

    # ## Lightning Trainer callbacks, loggers, plugins ## #

    # set logger
    csvlogger = CSVLogger(
        save_dir=log_dir,
        name="csv-logs",
    )
    #logger=Logger()

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
        #logger=[csvlogger],
        callbacks=callbacks,
        log_every_n_steps=50,
        #strategy=ray.train.lightning.RayDDPStrategy(find_unused_parameters=True),
        #plugins=[ray.train.lightning.RayLightningEnvironment()],
        #callbacks=[ray.train.lightning.RayTrainReportCallback(), LearningRateMonitor(logging_interval='epoch')],
        # [1a] Optionally, disable the default checkpointing behavior
        # in favor of the `RayTrainReportCallback` above.
        #enable_checkpointing=False,
    )
    hyperparameters = dict(lr=lr)
    lit_trainer.logger.log_hyperparams(hyperparameters)
    lit_trainer.fit(model=lit_module, datamodule=lit_datamodule)
    return lit_trainer.callback_metrics["val_f1"].item()

'''
Training without optuna
'''
'''
def train(
    accelerator: str = TrainerConfig.accelerator,  # Trainer flag
    devices: Union[int, str] = TrainerConfig.devices,  # Trainer flag
    strategy: str = TrainerConfig.strategy,  # Trainer flag
    precision: Optional[str] = TrainerConfig.precision,  # Trainer flag
    max_epochs: int = TrainerConfig.max_epochs,  # Trainer flag
    lr: float = ModuleConfig.learning_rate,  # learning rate for LightningModule
    batch_size: int = DataModuleConfig.batch_size,  # batch size for LightningDataModule DataLoaders
    perf: bool = False,  # set to True to log training time and other run information
    profile: bool = False,  # set to True to profile. only use profiler to identify bottlenecks
) -> None:
    """a custom Lightning Trainer utility

    Note:
        for all Trainer flags, see https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-flags
    """
    #Task.init(project_name="examples", task_name="BERT Disaster Tweets on Pytlightning")
    # ## LightningDataModule ## #
    lit_datamodule = AutoTokenizerDataModule(
        model_name=model_name,
        dataset_name=dataset_name,
        cache_dir=cache_dir,
        batch_size=batch_size,
    )

    # ## LightningModule ## #
    lit_module = CustomModel(learning_rate=lr)

    # ## Lightning Trainer callbacks, loggers, plugins ## #

    # set logger
    csvlogger = CSVLogger(
        save_dir=log_dir,
        name="csv-logs",
    )
    #logger=Logger()

    # set callbacks
    if perf:  # do not use EarlyStopping if getting perf benchmark
        callbacks = [
            ModelCheckpoint(
                dirpath=ckpt_dir,
                filename="model",
            ),
        ]
    else:
        callbacks = [
            #EarlyStopping(monitor="val_f1", mode="max", patience=3),
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
    if profile:
        profiler = PyTorchProfiler(dirpath=prof_dir)
    else:
        profiler = None

    # ## create Trainer and call .fit ## #
    lit_trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=precision,
        max_epochs=max_epochs,
        logger=[csvlogger, CometLogger(api_key="YOUR_COMET_API_KEY"), DVCLiveLogger(save_dvc_exp=True)],
        #logger=[csvlogger],
        callbacks=callbacks,
        profiler=profiler,
        log_every_n_steps=50,
        #strategy=ray.train.lightning.RayDDPStrategy(find_unused_parameters=True),
        #plugins=[ray.train.lightning.RayLightningEnvironment()],
        #callbacks=[ray.train.lightning.RayTrainReportCallback(), LearningRateMonitor(logging_interval='epoch')],
        # [1a] Optionally, disable the default checkpointing behavior
        # in favor of the `RayTrainReportCallback` above.
        #enable_checkpointing=False,
    )

    #trainer = ray.train.lightning.prepare_trainer(lit_trainer)
    start = perf_counter()
    lit_trainer.fit(model=lit_module, datamodule=lit_datamodule)

    #trainer.fit(model=lit_module, datamodule=lit_datamodule)
    stop = perf_counter()

    # ## log perf results ## #
    if perf:
        log_perf(start, stop, perf_dir, lit_trainer)

def train_warrper(train_config: dict = None) -> None:
    scaling_config = ray.train.ScalingConfig(num_workers=1, use_gpu=True)
    torch_config = TorchConfig(backend="gloo")
    if train_config is None:
        train_config = {}
        
    trainer = TorchTrainer(
        train,
        scaling_config=scaling_config,
        torch_config=torch_config,
        # [3a] If running in a multi-node cluster, this is where you
        # should configure the run's persistent storage that is accessible
        # across all worker nodes.
        # run_config=ray.train.RunConfig(storage_path="s3://..."),
    )
    result: ray.train.Result = trainer.fit()

'''
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
    study.optimize(objective, n_trials=200, timeout=86400) # seconds

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))