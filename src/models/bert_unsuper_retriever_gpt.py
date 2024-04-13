import os
import re
import hydra
from typing import Any, List

import logging
logging.basicConfig(level=logging.INFO)

import torch
import transformers
from omegaconf import DictConfig

# from .backbone.bert_retriever import BertRetrieval 

class SequenceClassification(torch.nn.Module):
    def __init__(
        self,
        num_labels =0,
        **kwargs: dict,
    ):
        super().__init__()

        self.tokenizer = kwargs.pop('tokenizer',None)
        self.config = DictConfig(kwargs)

        retriever = hydra.utils.get_class(self.config.retriever._class_) ## src.models.backbone.bert_retriever.BertRetrieval 
        self.retriever = retriever.from_pretrained(self.config.retriever.path, num_labels=num_labels, retriever=self.config.retriever, tokenizer=self.tokenizer['retriever'])
        generator = hydra.utils.get_class(self.config.generator._class_) ## transformers.GPT2LMHeadModel 
        self.generator = generator.from_pretrained(self.config.generator.path)
        self.retriever.train()
        self.generator.eval()

        self.loss_func = self.set_loss_func()
        self.acc_func = self.set_acc_func()

    def set_loss_func(self):
        def retriever_loss_func(output, labels, predict_text, reference):
            ## logits
            score = output['relevance_score']
            ## make labels
            ref = [len(self.extract_labels(d)) for d in reference]
            preds = [self.extract_labels(d, index=ref[i])[0] for i,d in enumerate(predict_text)]
            kbs_labels = [[int(g in p)]*score.shape[1] for p,g in zip(preds, labels)]
            kbs_labels = torch.tensor(kbs_labels).type(torch.FloatTensor).to(score.device)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(score, kbs_labels) 
            return loss
        return retriever_loss_func

    @torch.no_grad()
    def set_acc_func(self):
        def acc_func(predict_text, labels, reference):
            ref = [len(self.extract_labels(d)) for d in reference]
            preds = [self.extract_labels(d, index=ref[i])[0] for i,d in enumerate(predict_text)]
            acc = [int(g in p) for g,p in zip(labels, preds)]
            acc = torch.tensor(sum(acc)/len(acc))
            return acc
        return acc_func

    def forward(self,
            input_ids: torch.Tensor =None,
            labels: torch.Tensor =None,
            **kwargs):

        retriever_outputs = self.retriever(input_ids=input_ids)
        retrieved_doc = [' '.join(d) for d in retriever_outputs.concat_inputs['kbs']]

        generator_inputs_string = [f"{r} {i}" for r,i in zip(retrieved_doc, retriever_outputs.concat_inputs['inputs'])]
        print(generator_inputs_string[:3])
        generator_inputs = self.tokenizer['generator']( generator_inputs_string,
                                                        max_length = self.config.max_source_length + self.config.max_kb_length,
                                                        padding='max_length',
                                                        truncation='only_first',
                                                        return_tensors='pt' )
        print(self.tokenizer['generator'].batch_decode(generator_inputs['input_ids'])[:3])
        exit()
        
        generator_inputs = self.tokenizer['generator']( retrieved_doc, retriever_outputs.concat_inputs['inputs'],
                                                        max_length = self.config.max_source_length + self.config.max_kb_length,
                                                        padding='max_length',
                                                        truncation='only_first',
                                                        return_tensors='pt' )
        for key in generator_inputs:
            generator_inputs[key] = generator_inputs[key].to(self.generator.device)

        generator_outputs = self.generator.generate(**generator_inputs,
                                                    num_beams=self.config.generator.num_beams,
                                                    repetition_penalty=self.config.generator.repetition_penalty,
                                                    temperature=self.config.generator.temperature,
                                                    max_length=self.config.max_target_length)
        
        output = {
                'retriever_outputs': retriever_outputs,
                'generator_inputs': generator_inputs,
                'generator_outputs': generator_outputs,
                }
        return output

    def extract_labels(self, data, index=None):
        result = re.findall('< *y *>[^<]*< */ *y *>', data)
        try:
            if index != None: result = [result[index]]
        except IndexError as e:
            result = ['']
        result = [re.sub('^< *y *>','',d.strip()) for d in result]
        result = [re.sub('< */ *y *>$','',d.strip()) for d in result]
        result = [d.strip() for d in result]
        return result

    def training_step(self, batch: Any, batch_idx: int):
        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        for key in batch:
            if type(batch[key]) == list: continue
            batch[key] = batch[key].to(self.retriever.device)
        output = self.forward(**batch)

        retriever_outputs = output['retriever_outputs']
        logits = retriever_outputs['relevance_score']
        predict_text = self.tokenizer['generator'].batch_decode(output['generator_outputs'], skip_special_tokens=True)
        generator_inputs = self.tokenizer['generator'].batch_decode(output['generator_inputs']['input_ids'], skip_special_tokens=True)

        loss = self.loss_func(retriever_outputs, batch['labels'], predict_text, generator_inputs)
        acc = self.acc_func(predict_text, batch['labels'], generator_inputs)

        return {"loss": loss, "logits":logits, "acc":acc}

    def validation_step(self, batch: Any, batch_idx: int):
        for key in batch:
            if type(batch[key]) == list: continue
            batch[key] = batch[key].to(self.retriever.device)

        with torch.no_grad():
            output = self.forward(**batch)

            retriever_outputs = output['retriever_outputs']
            logits = retriever_outputs['relevance_score']
            predict_text = self.tokenizer['generator'].batch_decode(output['generator_outputs'], skip_special_tokens=True)
            generator_inputs = self.tokenizer['generator'].batch_decode(output['generator_inputs']['input_ids'], skip_special_tokens=True)

            loss = self.loss_func(retriever_outputs, batch['labels'], predict_text, generator_inputs)

        acc = self.acc_func(predict_text, batch['labels'], generator_inputs)
        
        return {"loss": loss, "logits":logits, "acc":acc}

    def test_step(self, batch: Any, batch_idx: int):
        raise NotImplementedError("test_step is not used.")

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
    
    def get_tokenizer(self):
        return self.tokenizer

    def predict_step(self, batch: Any, batch_idx: int):

        for key in batch:
            if type(batch[key]) == list: continue
            batch[key] = batch[key].to(self.retriever.device)

        input_ids = batch["input_ids"]

        with torch.no_grad():
            output = self.forward(input_ids)

            retriever_outputs = output['retriever_outputs']
            logits = retriever_outputs['relevance_score']
            generator_inputs = self.tokenizer['generator'].batch_decode(output['generator_inputs']['input_ids'], skip_special_tokens=True)
            predict_text = self.tokenizer['generator'].batch_decode(output['generator_outputs'], skip_special_tokens=True)
            loss = self.loss_func(retriever_outputs, batch['labels'], predict_text, generator_inputs)

        retriever_outputs = output['retriever_outputs']
        retrieved_labels = [[self.extract_labels(y) for y in x] for x in retriever_outputs['concat_inputs']['kbs']]
        ref = [self.extract_labels(d) for d in generator_inputs]
        predicts = [self.extract_labels(d, index=len(ref[i]))[0] for i,d in enumerate(predict_text)]
        labels = batch['labels']

        inputs = retriever_outputs['concat_inputs']['inputs']
        retrieved_doc = retriever_outputs['concat_inputs']['kbs']
        columns = ['inputs','preds','labels','generator_output','retrieved_doc','retrieved_labels','generator_inputs']
        result = zip(inputs, predicts, labels, predict_text, retrieved_doc, retrieved_labels, generator_inputs)
        result = [dict(zip(columns, d)) for d in result]
        return result


    def configure_optimizers(self, lr=1e-3):
        optimizer = torch.optim.Adam(self.retriever.parameters(), lr=lr)
        return optimizer

    def get_model(self):
        return self.retriever, self.generator

    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        if torch.cuda.device_count() > 1:
            raise NotImplementedError("Need model save code for multi-GPU.")
            #self.model.module.save_pretrained(path)
        else:
            self.retriever.save_pretrained(path)
        logging.info(f"SAVE {path}")

