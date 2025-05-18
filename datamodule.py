import os
import shutil
from datetime import datetime

from pathlib import Path
from typing import Union

from datasets import load_dataset
from torch.utils.data import DataLoader, default_collate
from transformers import AutoTokenizer

import lightning.pytorch as pl
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

from config import Config, DataModuleConfig, ModuleConfig


class AutoTokenizerDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: str = DataModuleConfig.dataset_name,
        cache_dir: Union[str, Path] = Config.cache_dir,
        model_name: str = ModuleConfig.model_name,
        max_length: int = ModuleConfig.max_length,
        num_labels: int = DataModuleConfig.num_classes,
        columns: list = ["input_ids", "attention_mask", "label", "token_type_ids"],
        batch_size: int = DataModuleConfig.batch_size,
        train_size: float = DataModuleConfig.train_size,
        stratify_by_column: str = DataModuleConfig.stratify_by_column,
        train_split: str = DataModuleConfig.train_split,
        test_split: str = DataModuleConfig.test_split,
        num_workers: int = DataModuleConfig.num_workers,
        seed: int = Config.seed,
    ) -> None:
        """a custom PyTorch Lightning LightningDataModule to tokenize text datasets

        Args:
            dataset_name: the name of the dataset as given on HF datasets
            cache_dir: corresponds to HF datasets.load_dataset
            model_name: the name of the model and accompanying tokenizer
            num_labels: the number of labels
            columns: the list of column names to pass to the HF dataset's .set_format method
            batch_size: the batch size to pass to the PyTorch DataLoaders
            train_size: the size of the training data split to pass to .train_test_split
            stratify_by_column: column name of labels to be used to perform stratified split of data
            train_split: the name of the training split as given on HF Hub
            test_split: the name of the test split as given on HF Hub
            num_workers: corresponds to torch.utils.data.DataLoader
            seed: the seed used in lightning.pytorch.seed_everything

        Notes:

        """
        super().__init__()

        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.model_name = model_name
        self.max_length = max_length
        self.train_size = train_size
        self.train_split = train_split
        self.test_split = test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.num_labels = num_labels
        self.columns = columns
        self.stratify_by_column = stratify_by_column

    def clear_custom_cache(self):
        """Custom method to clear cache"""
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)  # Remove the directory

    def prepare_data(self) -> None:
        """
        Notes:
            https://lightning.ai/docs/pytorch/stable/data/datamodule.html#prepare-data
        """
        pl.seed_everything(self.seed)
        # disable parrelism to avoid deadlocks
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        if not os.path.isdir(self.cache_dir):
            os.mkdir(self.cache_dir)

        cache_dir_is_empty = len(os.listdir(self.cache_dir)) == 0

        if cache_dir_is_empty:
            rank_zero_info(f"[{str(datetime.now())}] Downloading dataset.")
            load_dataset(self.dataset_name, cache_dir=self.cache_dir, use_auth_token=True)
        else:
            rank_zero_info(
                f"[{str(datetime.now())}] Data cache exists. Loading from cache."
            )

    def setup(self, stage: str) -> None:
        """
        Notes:
            https://lightning.ai/docs/pytorch/stable/data/datamodule.html#setup
        """
        if stage == "fit" or stage is None:
            # load and split
            dataset = load_dataset(
                self.dataset_name, cache_dir=self.cache_dir
            )
            print(dataset)
            #dataset = dataset.train_test_split(
            #    train_size=self.train_size, stratify_by_column=self.stratify_by_column
            #)
            # prep train
            self.train_data = dataset["train"].map(
                lambda example: {'text': example['text'], 
                                'label': list(example.values())[2:]},
                batched=False
            )            
            self.train_data = self.train_data.map(
                tokenize_text,
                batched=True,
                batch_size=1024,
                fn_kwargs={"model_name": self.model_name, "cache_dir": self.cache_dir, "max_length": self.max_length},
            )
            
            self.train_data.set_format("torch", columns=self.columns, output_all_columns=True)            
            # prep val
            self.val_data = dataset["test"].map(
                lambda example: {'text': example['text'], 
                                'label': list(example.values())[2:]},
                batched=False
            )
            
            self.val_data = self.val_data.map(
                tokenize_text,
                batched=True,
                batch_size=1024,
                fn_kwargs={"model_name": self.model_name, "cache_dir": self.cache_dir, "max_length": self.max_length},
            )

            self.val_data.set_format("torch", columns=self.columns)            
            # free mem from unneeded dataset obj
            del dataset
        if stage == "test" or stage is None:
            self.test_data = load_dataset(
                self.dataset_name, split=self.test_split, cache_dir=self.cache_dir
            )
            self.test_data.map(
                tokenize_text,
                batched=True,
                batch_size=512,
                fn_kwargs={"model_name": self.model_name, "cache_dir": self.cache_dir, "max_length": self.max_length},
            )
            self.test_data.set_format("torch", columns=self.columns)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """
        Notes:
            https://lightning.ai/docs/pytorch/stable/data/datamodule.html#train-dataloader
        """
        return DataLoader(
            self.train_data,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=lambda batch: default_collate([item for item in batch if item is not None])

        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """
        Notes:
            https://lightning.ai/docs/pytorch/stable/data/datamodule.html#val-dataloader
        """
        return DataLoader(
            self.val_data,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """
        Notes:
            https://lightning.ai/docs/pytorch/stable/data/datamodule.html#test-dataloader
        """
        return DataLoader(
            self.test_data,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )


def tokenize_text(
    batch: dict,
    *,
    model_name: str,
    cache_dir: Union[str, Path],
    truncation: bool = True,  # leave as True if dataset has sequences that exceed the model's max sequence length
    padding: bool = "max_length",  # pad so that all tensors are of the same dimensions
    max_length: int = 512,
):
    """
    Notes:
        https://huggingface.co/docs/transformers/v4.38.2/en/model_doc/auto#transformers.AutoTokenizer.from_pretrained
    """
    #print(batch["text"])
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    text = batch if isinstance(batch, str) else batch["text"]  # allow for inference input as raw text
    if isinstance(text, list):
        text = [str(t) for t in text]
    else:
        text = str(text)
    final_token = tokenizer(text, truncation=truncation, padding=padding, return_tensors="pt", return_token_type_ids=True, max_length=max_length)
    #print(tokenizer.model_max_length)
    '''
    for i in range(len(final_token['input_ids'])):
                assert len(final_token['input_ids'][i]) == max_length, f"input_ids length mismatch at index {i}, expected {max_length} but got {len(final_token['input_ids'][i])}"
                assert len(final_token['attention_mask'][i]) == max_length, f"attention_mask length mismatch at index {i}, expected {max_length} but got {len(final_token['attention_mask'][i])}"
                if 'token_type_ids' in final_token:
                    assert len(final_token['token_type_ids'][i]) == max_length, f"token_type_ids length mismatch at index {i}, expected {max_length} but got {len(final_token['token_type_ids'][i])}"
    #assert len(segment_ids) == max_seq_length
    '''
    
    return final_token
