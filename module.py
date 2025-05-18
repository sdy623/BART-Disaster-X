from abc import abstractmethod
from pathlib import Path
from typing import Optional, Union, Tuple, List

import torch
import numpy as np
from transformers import BertForSequenceClassification, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import get_cosine_schedule_with_warmup


import lightning.pytorch as pl
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from torchmetrics.functional import accuracy
from torchmetrics.classification import MultilabelAccuracy, MultilabelPrecision, MultilabelRecall, MultilabelF1Score

from datamodule import tokenize_text
from config import Config, DataModuleConfig, ModuleConfig
import torch.nn as nn
import torch.optim as optim

from peft import LoraConfig, TaskType, get_peft_model

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return max(lr_factor, 2e-6)
    
class FocalLoss(nn.Module):
    """
    Focal loss implemented as a PyTorch module.
    [Original paper](https://arxiv.org/pdf/1708.02002.pdf).
    """

    def __init__(self, gamma: float, reduction='none'):
        """
        :param gamma: What value of Gamma to use. Value of 0 corresponds to Cross entropy.
        :param reduction: Reduction to be done on top of datapoint-level losses.
        """
        super().__init__()

        assert reduction in ['none', 'sum', 'mean']

        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_logits: torch.Tensor, targets: torch.Tensor):
        ce_loss = torch.nn.functional.cross_entropy(input_logits, targets, reduction='none')
        input_probs_for_target = torch.exp(-ce_loss)
        loss = (1 - input_probs_for_target) ** self.gamma * ce_loss

        if self.reduction == 'sum':
            loss = loss.sum(dim=-1)
        elif self.reduction == 'mean':
            loss = loss.mean(dim=-1)

        return loss
    
@abstractmethod
class EncoderBase(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size

class BERTEmbeeding(EncoderBase):
    def __init__(self, model_name,
                 attention_probs_dropout: Optional[float] = None):
        super().__init__(model_name)
        assert attention_probs_dropout is None or 0 <= attention_probs_dropout <= 1, \
            "attention_probs_dropout must be between 0 and 1 or None"
        self.hidden_size = self.encoder.config.hidden_size
        if attention_probs_dropout:
            self.encoder = AutoModel.from_pretrained(
                model_name,
                attention_probs_dropout_prob=attention_probs_dropout
            )
    def forward(self, input_ids, 
                attention_mask=None, 
                token_type_ids=None,
                position_ids=None,
                head_mask=None,):
        
        outputs = self.encoder(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask,)
        return outputs.last_hidden_state
    

class ModernBERTEmbeeding(EncoderBase):
    def __init__(self, model_name,):
        super().__init__(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.encoder = AutoModel.from_pretrained(
            model_name,
        )
    def forward(self, input_ids, 
                attention_mask=None, 
                position_ids=None,):
        
        outputs = self.encoder(input_ids=input_ids,
                               attention_mask=attention_mask,
                               position_ids=position_ids,)
        return outputs.last_hidden_state

'''
The accuracy of the NV-Embed model is not as good as the BERT model, 
and it is very slow occupying 16GB+ of GPU memory.
'''
class Nvembeddings(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        model = AutoModel.from_pretrained("nvidia/NV-Embed-v2", trust_remote_code=True).half().cuda()
        lora_config = LoraConfig(
                r=128,
                lora_alpha=32,
                target_modules=["to_q", "to_kv"],
                lora_dropout=0.08,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION
                )
        self.encoder = get_peft_model(model, lora_config)
        # self.encoder = model
        self.hidden_size = 4096
    def forward(self, input_ids, 
            attention_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs,):
        outputs = self.encoder(input_ids=input_ids,
                            attention_mask=attention_mask,
                            inputs_embeds=inputs_embeds,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states,
                            return_dict=return_dict,
                            **kwargs,)
        return outputs['sentence_embeddings']
    
class BARTEmbeddings(EncoderBase):
    def __init__(self, model_name, 
                 attention_dropout: Optional[float] = None):
        super().__init__(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.encoder = AutoModel.from_pretrained(model_name)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        
        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return outputs.last_hidden_state
    
class ClassifierBase(nn.Module):
    def __init__(self, input_dim, num_labels):
        super().__init__()
        self.input_dim = input_dim
        self.num_labels = num_labels

    def forward(self, hidden_states):
        raise NotImplementedError("This method should be overridden in subclasses")

class BertLinearClassificationHead(ClassifierBase):
    """Head for sentence-level classification tasks."""
    def __init__(
            self,
            input_dim: int,
            num_labels: int,
            opposing_label_sets: List[Tuple[int, int]] = None,
            classifier_dropout: Optional[float] = None
    ):
        assert classifier_dropout is None or 0 <= classifier_dropout <= 1, \
            "pooler_dropout must be between 0 and 1 or None"
        super().__init__(input_dim, num_labels)
        self.opposing_label_sets = opposing_label_sets  # List of tuples with opposing label indices
        self.dropout = nn.Dropout(p=classifier_dropout) if classifier_dropout else None
        self.linear = nn.Linear(input_dim, num_labels)

    def forward(self, hidden_states):
        #pooled_output = torch.mean(hidden_states, dim=1)
        cls_output = hidden_states[:, 0, :]
        if self.dropout:
            cls_output = self.dropout(cls_output)
        logits = self.linear(cls_output)

        # Apply Softmax to opposing labels
        if self.opposing_label_sets is not None:
            for label_set in self.opposing_label_sets:
                logits[:, label_set] = torch.softmax(logits[:, label_set], dim=1).to(logits.dtype)

        # Apply Sigmoid to all logits for multi-label outputs
        return logits

class ClassificationHEAD(ClassifierBase):
    def __init__(self, input_dim, num_labels, opposing_label_sets: List[Tuple[int, int]]=None):
        super().__init__(input_dim, num_labels)
        self.opposing_label_sets = opposing_label_sets  # List of tuples with opposing label indices
        self.linear = nn.Linear(input_dim, num_labels)

    def forward(self, hidden_states):
        #pooled_output = torch.mean(hidden_states, dim=1)
        cls_output = hidden_states[:, 0, :]
        logits = self.linear(cls_output)

        # Apply Softmax to opposing labels
        if self.opposing_label_sets is not None:
            for label_set in self.opposing_label_sets:
                logits[:, label_set] = torch.softmax(logits[:, label_set], dim=1)
        
        # Apply Sigmoid to all logits for multi-label outputs
        return logits

class LSTMAttnClassificationHEAD(ClassifierBase):
    def __init__(self, input_dim, num_labels, opposing_label_sets: List[Tuple[int, int]]=None):
        super().__init__(input_dim, num_labels)
        self.opposing_label_sets = opposing_label_sets  # List of tuples with opposing label indices
        self.lstm = nn.LSTM(input_dim, input_dim, batch_first=True,)
        self.linear = nn.Linear(input_dim*2, num_labels)
        self.multiAttn = nn.MultiheadAttention(input_dim, input_dim)

    def forward(self, output):
        out, _ = self.lstm(output[0], None)
        query = key = value = out.permute(1, 0, 2)
        attn_output, weights = self.multiAttn(query, key, value)
        sequence_output = out[:, -1, :]
        attn_output = attn_output.permute(1, 0, 2)
        

        pooled_output = attn_output.mean(dim=1) 
        input = torch.cat([pooled_output, output[1]], dim=-1)
        logits = self.linear(input)
        
        # Apply Softmax to opposing labels
        if self.opposing_label_sets is not None:
            for label_set in self.opposing_label_sets:
                logits[:, label_set] = torch.softmax(logits[:, label_set], dim=1)
        
        # Apply Sigmoid to all logits for multi-label outputs
        return logits

class LSTMClassificationHEAD(ClassifierBase):
    def __init__(self, input_dim, num_labels, opposing_label_sets: List[Tuple[int, int]]=None):
        super().__init__(input_dim, num_labels)
        self.opposing_label_sets = opposing_label_sets  # List of tuples with opposing label indices
        self.lstm = nn.LSTM(input_dim, input_dim, batch_first=True)
        self.linear = nn.Linear(input_dim, num_labels)
        #self.multiAttn = nn.MultiheadAttention(input_dim, input_dim)

    def forward(self, last_hidden_state):
        out, _ = self.lstm(last_hidden_state, None)
        
        sequence_output = out[:, -1, :]
        # pooled_output = torch.mean(out, dim=1)
        #cls_output = hidden_states[:, 0, :]
        logits = self.linear(sequence_output)
        # logits = self.linear(pooled_output)
        # Apply Softmax to opposing labels
        '''
        if self.opposing_label_sets is not None:
            for label_set in self.opposing_label_sets:
                logits[:, label_set] = torch.softmax(logits[:, label_set], dim=1)
        '''
        # Apply Sigmoid to all logits for multi-label outputs
        return logits

class CustomClassifyModelCommon(pl.LightningModule):
    def __init__(self,
        model_name: str = ModuleConfig.model_name,
        num_classes: int = DataModuleConfig.num_classes,  # set according to the finetuning dataset
        input_key: str = "input_ids",  # set according to the finetuning dataset
        label_key: str = "label",  # set according to the finetuning dataset
        mask_key: str = "attention_mask",  # set according to the model output object
        output_key: str = "logits",  # set according to the model output object
        loss_key: str = "loss",  # set according to the model output object
        attention_probs_dropout: Optional[float] = ModuleConfig.attention_probs_dropout,
        classifier_dropout: Optional[float] = ModuleConfig.classifier_dropout,
        learning_rate: float = ModuleConfig.learning_rate,
        learning_rate_bert: float = ModuleConfig.learning_rate_bert,
        learning_rate_lstm: float = ModuleConfig.learning_rate_lstm,
        lr_gamma: float = 0.76825,
        opposing_label_sets: List[Tuple[int, int]] = None,
        warmup: int = ModuleConfig.warming_steps,):

        super().__init__()
        self.save_hyperparameters()

        self.input_key = input_key
        self.label_key = label_key
        self.mask_key = mask_key
        self.output_key = output_key
        self.loss_key = loss_key
        self.num_classes = num_classes
        self.lr_gamma= lr_gamma
          
        self.learning_rate = learning_rate
        self.learning_rate_bert = learning_rate_bert
        self.learning_rate_lstm = learning_rate_lstm

        self.opposing_label_sets = opposing_label_sets
        self.criterion = nn.BCEWithLogitsLoss()  # Use BCEWithLogits for multi-label setting
        #self.criterion = FocalLoss(gamma=ModuleConfig.focal_gamma, reduction='sum')
        # Metrics
        self.accuracy = MultilabelAccuracy(num_labels=num_classes, average='micro')
        self.precision = MultilabelPrecision(num_labels=num_classes, average='micro')
        self.recall = MultilabelRecall(num_labels=num_classes, average='micro')
        self.f1_score = MultilabelF1Score(num_labels=num_classes, average='micro')
        self.macro_f1_score = MultilabelF1Score(num_labels=num_classes, average='macro')

    def neo_criterion(self, logits, targets):
        total_loss = 0

        # Can set opposing label sets
        if self.opposing_label_sets:
            all_opposing_labels = []
            for label_set in self.opposing_label_sets:

                all_opposing_labels.extend(label_set)

                logits_subset = logits[:, label_set]  # [batch_size, num_labels_in_set]
                targets_subset = targets[:, label_set]  # [batch_size, num_labels_in_set]

                targets_indices = torch.argmax(targets_subset, dim=1)

                ce_loss_fn = nn.CrossEntropyLoss()
                ce_loss = 0.4 * ce_loss_fn(logits_subset, targets_indices)
                total_loss += ce_loss

        else:
            all_opposing_labels = []

        # Non opposing labels
        non_opposing_labels = [i for i in range(self.num_classes) if i not in all_opposing_labels]
        if non_opposing_labels:
            logits_non_opposing = logits[:, non_opposing_labels]
            targets_non_opposing = targets[:, non_opposing_labels]

            # Calculate BCE loss
            bce_loss_fn = nn.BCEWithLogitsLoss()
            bce_loss = 0.6 * bce_loss_fn(logits_non_opposing, targets_non_opposing)
            total_loss += bce_loss

        return total_loss
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        hidden_states = self.encoder(input_ids, attention_mask)
        logits = self.classifier(hidden_states)
        return logits

    def training_step(self, batch, batch_idx):
        logits = self(input_ids=batch[self.input_key], attention_mask=batch[self.mask_key])
        loss = self.criterion(logits, batch[self.label_key].float())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(input_ids=batch[self.input_key], attention_mask=batch[self.mask_key])
        loss = self.criterion(logits, batch[self.label_key].float())
        # preds = (torch.sigmoid(logits) > 0.5).int() # Without handling opposing labels
        
        all_opposing_labels = []
        if self.opposing_label_sets is not None:
            for label_set in self.opposing_label_sets:
                all_opposing_labels.extend(label_set)
                logits[:, label_set] = torch.softmax(logits[:, label_set].float(), dim=1).to(logits.dtype)

        non_opposing_labels = [i for i in range(self.num_classes) if i not in all_opposing_labels]
        if non_opposing_labels:
            logits[:, non_opposing_labels] = torch.sigmoid(logits[:, non_opposing_labels])
        
        preds = (logits > 0.5).int()

        # Calculate metrics
        acc = self.accuracy(preds, batch[self.label_key])
        prec = self.precision(preds, batch[self.label_key])
        rec = self.recall(preds, batch[self.label_key])
        f1 = self.f1_score(preds, batch[self.label_key])
        marco_f1 = self.macro_f1_score(preds, batch[self.label_key])

        # Log metrics
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_accuracy", acc, on_epoch=True)
        self.log("val_precision", prec, on_epoch=True)
        self.log("val_recall", rec, on_epoch=True)
        self.log("val_f1", f1, on_epoch=True)
        self.log("val_macro_f1", marco_f1, on_epoch=True)

        return {"val_loss": loss, "val_accuracy": acc, "val_precision": prec, "val_recall": rec, "val_f1": f1, "val_macro_f1": marco_f1,}
    
    def predict_step(
        self, sequence: str, threshold: float = 0.5, cache_dir: Union[str, Path] = Config.cache_dir
        ) -> str:
        # Tokenize the input sequence
        inputs = tokenize_text(sequence, self.input_key, self.mask_key, cache_dir=cache_dir)
        # Move inputs to the same device as the model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # Forward pass
        logits = self(input_ids=inputs[self.input_key], attention_mask=inputs[self.mask_key])
        # Apply sigmoid activation to get probabilities
        probs = torch.sigmoid(logits)
        # Apply threshold to get binary predictions
        preds = (probs > threshold).int()
        # Convert predictions to a list of labels
        labels = [i for i, pred in enumerate(preds[0]) if pred == 1]
        # Convert labels to a string
        labels_str = ", ".join([str(label) for label in labels])
        return labels_str


    def configure_optimizers(self) -> OptimizerLRScheduler:
        
        bert_params = [param for name, param in self.named_parameters() if "bert" in name]
        lstm_params = [param for name, param in self.named_parameters() if "lstm" in name or "linear" in name]
        
        
        # Training with different learning rates for BERT and LSTM
        '''
        optimizer = torch.optim.AdamW([
            {"params": bert_params, "lr": self.learning_rate_bert},
            {"params": lstm_params, "lr": self.learning_rate_lstm}
        ])
        '''
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)


        self.lr_scheduler = CosineWarmupScheduler(
            optimizer, warmup=self.hparams.warmup, max_iters=self.trainer.estimated_stepping_batches
        )
        
        fixed_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=3)

        # [Disabled] Define the cosine annealing learning rate scheduler for the remaining epochs
        # cosine_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=5e-7)

        reduce_lr_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.35, patience=3, verbose=True)
        # [Disabled] Create the exp scheduler
        # exp_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.lr_gamma)
        # [Disabled] Combine the schedulers
        # combined_scheduler = torch.optim.lr_scheduler.ChainedScheduler(schedulers=[fixed_lr_scheduler, exp_scheduler])

        return optimizer

    
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler.step()  # Step per iteration
        '''
    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        '''
        
class ModernBERTModel(CustomClassifyModelCommon):
    def __init__(self,
        model_name: str = ModuleConfig.model_name,
        num_classes: int = DataModuleConfig.num_classes,  # set according to the finetuning dataset
        input_key: str = "input_ids",  # set according to the finetuning dataset
        label_key: str = "label",  # set according to the finetuning dataset
        mask_key: str = "attention_mask",  # set according to the model output object
        output_key: str = "logits",  # set according to the model output object
        loss_key: str = "loss",  # set according to the model output object
        attention_probs_dropout: Optional[float] = ModuleConfig.attention_probs_dropout,
        classifier_dropout: Optional[float] = ModuleConfig.classifier_dropout,
        learning_rate: float = ModuleConfig.learning_rate,
        learning_rate_bert: float = ModuleConfig.learning_rate_bert,
        learning_rate_lstm: float = ModuleConfig.learning_rate_lstm,
        lr_gamma: float = 0.76825,
        opposing_label_sets: List[Tuple[int, int]] = None,
        warmup: int = ModuleConfig.warming_steps,):
        
        super().__init__()
        self.encoder = ModernBERTEmbeeding(model_name)
        self.classifier = LSTMClassificationHEAD(self.encoder.hidden_size, num_classes, opposing_label_sets)
        
    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.encoder(input_ids, attention_mask)
        logits = self.classifier(hidden_states)
        return logits
    
class CustomModel(pl.LightningModule):
    def __init__(self,
        model_name: str = ModuleConfig.model_name,
        num_classes: int = DataModuleConfig.num_classes,  # set according to the finetuning dataset
        input_key: str = "input_ids",  # set according to the finetuning dataset
        label_key: str = "label",  # set according to the finetuning dataset
        mask_key: str = "attention_mask",  # set according to the model output object
        output_key: str = "logits",  # set according to the model output object
        loss_key: str = "loss",  # set according to the model output object
        attention_probs_dropout: Optional[float] = ModuleConfig.attention_probs_dropout,
        classifier_dropout: Optional[float] = ModuleConfig.classifier_dropout,
        learning_rate: float = ModuleConfig.learning_rate,
        learning_rate_bert: float = ModuleConfig.learning_rate_bert,
        learning_rate_lstm: float = ModuleConfig.learning_rate_lstm,
        lr_gamma: float = 0.76825,
        opposing_label_sets: List[Tuple[int, int]] = None,
        warmup: int = ModuleConfig.warming_steps,):

        super().__init__()

        self.encoder = BARTEmbeddings(model_name, attention_probs_dropout)
        self.classifier = LSTMClassificationHEAD(self.encoder.hidden_size, num_classes, opposing_label_sets)

class LinearBEModel(pl.LightningModule):
    def __init__(self,
        model_name: str = ModuleConfig.model_name,
        num_classes: int = DataModuleConfig.num_classes,  # set according to the finetuning dataset
        input_key: str = "input_ids",  # set according to the finetuning dataset
        label_key: str = "label",  # set according to the finetuning dataset
        mask_key: str = "attention_mask",  # set according to the model output object
        output_key: str = "logits",  # set according to the model output object
        loss_key: str = "loss",  # set according to the model output object
        attention_probs_dropout: Optional[float] = ModuleConfig.attention_probs_dropout,
        classifier_dropout: Optional[float] = ModuleConfig.classifier_dropout,
        learning_rate: float = ModuleConfig.learning_rate,
        learning_rate_bert: float = ModuleConfig.learning_rate_bert,
        learning_rate_lstm: float = ModuleConfig.learning_rate_lstm,
        lr_gamma: float = 0.76825,
        opposing_label_sets: List[Tuple[int, int]] = None,
        warmup: int = ModuleConfig.warming_steps,):

        super().__init__()
      
        self.encoder = BERTEmbeeding(model_name, attention_probs_dropout)
        self.classifier = BertLinearClassificationHead(self.encoder.hidden_size, 
                                                       num_classes, 
                                                       classifier_dropout=classifier_dropout, 
                                                       opposing_label_sets=opposing_label_sets)

        
        
class LinearBAModel(pl.LightningModule):
    def __init__(self,
        model_name: str = ModuleConfig.model_name,
        num_classes: int = DataModuleConfig.num_classes,  # set according to the finetuning dataset
        input_key: str = "input_ids",  # set according to the finetuning dataset
        label_key: str = "label",  # set according to the finetuning dataset
        mask_key: str = "attention_mask",  # set according to the model output object
        output_key: str = "logits",  # set according to the model output object
        loss_key: str = "loss",  # set according to the model output object
        attention_probs_dropout: Optional[float] = ModuleConfig.attention_probs_dropout,
        classifier_dropout: Optional[float] = ModuleConfig.classifier_dropout,
        learning_rate: float = ModuleConfig.learning_rate,
        learning_rate_bert: float = ModuleConfig.learning_rate_bert,
        learning_rate_lstm: float = ModuleConfig.learning_rate_lstm,
        lr_gamma: float = 0.76825,
        opposing_label_sets: List[Tuple[int, int]] = None,
        warmup: int = ModuleConfig.warming_steps,):

        super().__init__()
        self.save_hyperparameters()

        self.input_key = input_key
        self.label_key = label_key
        self.mask_key = mask_key
        self.output_key = output_key
        self.loss_key = loss_key
        self.num_classes = num_classes
        self.lr_gamma= lr_gamma
        
        self.encoder = BARTEmbeddings(model_name, attention_probs_dropout)
        self.classifier = BertLinearClassificationHead(self.encoder.hidden_size, 
                                                       num_classes, 
                                                       classifier_dropout=classifier_dropout, 
                                                       opposing_label_sets=opposing_label_sets)