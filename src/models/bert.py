import os
import hydra
from typing import Any, List

import logging
logging.basicConfig(level=logging.INFO)

import torch
import transformers
from omegaconf import DictConfig

class SequenceClassification(torch.nn.Module):
    def __init__(
        self,
        num_labels =0,
        **kwargs: dict,
    ):
        super().__init__()

        self.config = DictConfig(kwargs)

        self.tokenizer = None

        model = hydra.utils.get_class(self.config._class_) ## transformers.BertForSequenceClassification
        self.model = model.from_pretrained(self.config.path, num_labels=num_labels)
        self.model.train()

        self.pad_token_id = 0

        self.loss_func = self.set_loss_func()
        self.acc_func = self.set_acc_func()

    def set_loss_func(self):
        return torch.nn.CrossEntropyLoss()

    @torch.no_grad()
    def set_acc_func(self):
        def acc_func(preds, labels):
            preds = torch.argmax(preds, dim=1)
            acc = preds == labels
            acc = torch.sum(acc)/acc.size(0)
            return acc
        return acc_func

    def forward(self,
            input_ids: torch.Tensor =None,
            labels: torch.Tensor =None,
            **kwargs):

        output = self.model(input_ids=input_ids)
        return output

    def training_step(self, batch: Any, batch_idx: int):
        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        for key in batch:
            batch[key] = batch[key].to(self.model.device)
        output = self.forward(**batch)

        logits = output.logits
        loss = output.loss
        if loss is None:
            loss = self.loss_func(logits, batch['labels'])
        acc = self.acc_func(logits, batch['labels'])

        return {"loss": loss, "logits":logits, "acc":acc}

    def validation_step(self, batch: Any, batch_idx: int):
        for key in batch:
            batch[key] = batch[key].to(self.model.device)

        with torch.no_grad():
            output = self.forward(**batch)
            logits = output.logits
            loss = output.loss
            if loss is None:
                loss = self.loss_func(logits, batch['labels'])

            acc = self.acc_func(logits, batch['labels'])
        
        return {"loss": loss, "logits":logits, "acc":acc}

    def test_step(self, batch: Any, batch_idx: int):
        raise NotImplementedError("test_step is not used.")

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
    
    def get_tokenizer(self):
        return self.tokenizer

    def predict_step(self, batch: Any, batch_idx: int):

        for key in batch:
            batch[key] = batch[key].to(self.model.device)

        input_ids = batch["input_ids"]

        output = self.forward(input_ids)
        predict = torch.argmax(output.logits, dim=-1)

        labels = batch['labels']

        columns = ['inputs','preds','labels']
        result = zip(input_ids.tolist(), predict.tolist(), labels.tolist())
        result = [dict(zip(columns, d)) for d in result]
        return result


    def configure_optimizers(self, lr=1e-3):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        return optimizer

    def get_model(self):
        return self.model

    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        if torch.cuda.device_count() > 1:
            raise NotImplementedError("Need model save code for multi-GPU.")
            #self.model.module.save_pretrained(path)
        else:
            self.model.save_pretrained(path)
        logging.info(f"SAVE {path}")

