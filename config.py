import os

from dataclasses import dataclass, field
from multiprocessing import cpu_count
from pathlib import Path
from typing import Optional, Union, List, Tuple

# ## get root path ## #
'''
this_file = Path(__file__)
this_studio_idx = [
    i for i, j in enumerate(this_file.parents) if j.name.endswith("this_studio")
][0]
'''
this_studio = "./"

@dataclass
class Config:
    cache_dir: str = os.path.join(this_studio, "data")
    log_dir: str = os.path.join(this_studio, "logs")
    ckpt_dir: str = os.path.join(this_studio, "checkpoints")
    prof_dir: str = os.path.join(this_studio, "logs", "profiler")
    perf_dir: str = os.path.join(this_studio, "logs", "perf")
    seed: int = 59631546


@dataclass
class ModuleConfig:
    model_name = "answerdotai/ModernBERT-large"
    # model_name: str = "nvidia/NV-Embed-v2" # change this to use a different pretrained checkpoint and tokenizer
    learning_rate: float = 2e-5
    learning_rate_bert: float = 1.2e-5
    learning_rate_lstm: float = 7e-5
    finetuned: str = "checkpoints/twhin-bert-base-finetuned" # change this to use a different pretrained checkpoint and tokenizer
    max_length: int = 128
    attention_probs_dropout: float = 0.1
    classifier_dropout: Optional[float] = None
    warming_steps: int = 100
    focal_gamma: float = 2.0
    
@dataclass
class DataModuleConfig:
    dataset_name: str = "sdy623/new_disaster_tweets" # change this to use different dataset
    num_classes: int = 12 # change this to set the number of classes
    batch_size: int = 8 # change this to set the batch size
    train_split: str = "train"
    test_split: str = "test"
    train_size: float = 0.8
    stratify_by_column: str = "label"
    num_workers: int = 0
    

@dataclass
class TrainerConfig:
    accelerator: str = "auto" # Trainer flag
    devices: Union[int, str] = "auto"  # Trainer flag
    strategy: str = "auto"  # Trainer flag
    precision: Optional[str] = "bf16-mixed"  # Trainer flag
    max_epochs: int = 10  # Trainer flag