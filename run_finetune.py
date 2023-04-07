#!/usr/bin/env python
# coding=utf-8

import os
import time
import argparse
import json
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything, Trainer, LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from transformers import get_linear_schedule_with_warmup, AdamW, T5Tokenizer

from torchsummary import summary
from engine.evaluating import measuring, Evaluation
from engine.constants import ModelType, TaskType, LHESequenceMarker
from engine.module import StrcutT5ForConditionalGeneration, StrcutFTT5ForConditionalGeneration, T5ForConditionalGeneration
from engine.data_utils import MyDataset, loading_data
from engine.utils import get_logger

model_dict = {
    ModelType.UIE: T5ForConditionalGeneration,
    ModelType.LASUIE: StrcutT5ForConditionalGeneration,
    ModelType.LASUIE_STRUCT_TUNING: StrcutFTT5ForConditionalGeneration,
}


def get_model(model_type, lm_location):
    return model_dict[model_type].from_pretrained(lm_location)


class ModelWrapper(LightningModule):
    def __init__(self, dataset, model_type, task_type, lm_location, learning_rate, other_learning_rate, adam_epsilon, warmup_steps, weight_decay,
                 train_batch_size, val_batch_size, max_length, gradient_accumulation_steps, is_train=True):
        super(ModelWrapper, self).__init__()
        self.save_hyperparameters()
        self.model_type = model_type
        self.task_type = task_type
        self.lm_location = lm_location
        self.model = get_model(self.model_type, self.lm_location)
        self.tokenizer = T5Tokenizer.from_pretrained(self.lm_location)
        self.tokenizer.bos_token = LHESequenceMarker.seq_start

        self.loss_fn = nn.CrossEntropyLoss()

        self.to_remove_token_list = list()
        self.init_removal_token_list()

        self.is_train = is_train

    def init_removal_token_list(self):
        if self.tokenizer.unk_token:
            self.to_remove_token_list += [self.tokenizer.unk_token]
        if self.tokenizer.eos_token:
            self.to_remove_token_list += [self.tokenizer.eos_token]
        if self.tokenizer.pad_token:
            self.to_remove_token_list += [self.tokenizer.pad_token]

    def forward(self, batch):
        return self.model(input_ids=batch['input_ids'], attention_mask=batch['masks'], labels=batch['output_ids'])

    def generate(self, input_ids, attention_mask, num_beams=2, min_length=0, max_length=50):
        return self.model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=num_beams, min_length=min_length,
                                   max_length=max_length)

    def _step(self, batch):
        outputs = self(batch)
        loss = outputs[0]
        logits = outputs[1]

        return loss, logits

    def training_step(self, batch, batch_idx):
        loss, _ = self._step(batch)
        return loss

    def postprocess_text(self, x_str):
        for to_remove_token in self.to_remove_token_list:
            x_str = x_str.replace(to_remove_token, '')
        return x_str.strip()

    def validation_step(self, batch, batch_idx):
        loss, logits = self._step(batch)
        max_ids = torch.max(logits, dim=-1)
        preds = self.tokenizer.batch_decode(max_ids[1], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        golds = self.tokenizer.batch_decode(batch['output_ids'], skip_special_tokens=False, clean_up_tokenization_spaces=False)

        preds = [self.postprocess_text(x) for x in preds]
        golds = [self.postprocess_text(x) for x in golds]

        if self.is_train:
            result = measuring([Evaluation.ROUGE], preds, golds, self.task_type)
            return {'loss': loss.cpu().numpy(),
                    'rouge': result["rouge"],
                    }
        else:
            result = measuring([Evaluation.ROUGE, Evaluation.UIE], preds, golds, self.task_type)
            if self.task_type == TaskType.SPAN:
                return {
                    'rouge': result["rouge"],
                    'span-F1': result["span_F1"],
                }
            else:
                return {
                    'rouge': result["rouge"],
                    'span-F1': result["span_F1"],
                    'triplet-F1': result["triplet_F1"],
                }

    def validation_epoch_end(self, outputs):
        mylogger.info(f'current epoch: {self.current_epoch}')
        if self.is_train:
            avg_loss = np.stack([x['loss'] for x in outputs]).mean()
            avg_loss = round(float(avg_loss), 3)
            avg_rouge_score = np.stack([x['rouge'].cpu().numpy() for x in outputs]).mean()
            avg_rouge_score = round(float(avg_rouge_score), 4)
            self.log('rouge', avg_rouge_score)
            mylogger.info(f'validating loss: {avg_loss} || rouge: {avg_rouge_score}')
            return {'loss': avg_loss,
                    'rouge': avg_rouge_score,
                    }
        else:
            avg_rouge_score = np.stack([x['rouge'] for x in outputs]).mean()
            avg_span_F1 = np.stack([x['span-F1'] for x in outputs]).mean()
            avg_rouge_score = round(float(avg_rouge_score), 4)
            avg_span_F1 = round(float(avg_span_F1), 4)
            self.log('rouge', avg_rouge_score)
            if self.task_type == TaskType.SPAN:
                mylogger.info(f'rouge: {avg_rouge_score} || span-F1: {avg_span_F1}')
                return {'rouge': avg_rouge_score,
                        'span-F1': avg_span_F1,
                        }
            else:
                avg_triplet_F1 = np.stack([x['triplet-F1'] for x in outputs]).mean()
                avg_triplet_F1 = round(float(avg_triplet_F1), 4)
                mylogger.info(f'rouge: {avg_rouge_score} || span-F1: {avg_span_F1} || triplet-F1: {avg_triplet_F1}')
                return {'rouge': avg_rouge_score,
                        'span-F1': avg_span_F1,
                        'triplet-F1': avg_triplet_F1,
                        }

    def configure_optimizers(self):
        if self.model_type in [ModelType.LASUIE, ModelType.LASUIE_STRUCT_TUNING]:
            gat_params = list(set(self.model.ConGAT.parameters()).union(set(self.model.DepGAT.parameters())).union(set(self.model.HSI.parameters())))
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.hparams.weight_decay,
                    "lr": self.hparams.learning_rate,
                },
                {
                    "params": [p for n, p in self.model.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                    "lr": self.hparams.learning_rate,
                },
                {
                    "params": gat_params,
                    "weight_decay": self.hparams.weight_decay,
                    "lr": self.hparams.other_learning_rate,
                },
            ]
            optimizer = AdamW(optimizer_grouped_parameters, eps=self.hparams.adam_epsilon,
                              lr=self.hparams.other_learning_rate)
        elif self.model_type == ModelType.UIE:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.hparams.weight_decay,
                    "lr": self.hparams.learning_rate,
                },
                {
                    "params": [p for n, p in self.model.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                    "lr": self.hparams.learning_rate,
                }
            ]
            optimizer = AdamW(optimizer_grouped_parameters, eps=self.hparams.adam_epsilon,
                              lr=self.hparams.learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.hparams.warmup_steps,
                                                    num_training_steps=self.trainer.estimated_stepping_batches)
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default='re', type=str)
    parser.add_argument("--task_type", default='pair', help="span, pair, hyperpair", type=str)
    parser.add_argument("--data", default='nyt', type=str)

    parser.add_argument("--train_file", default='train.json', type=str)
    parser.add_argument("--eval_file", default='dev.json', type=str)
    parser.add_argument("--test_file", default='test.json', type=str)
    parser.add_argument("--label_file", default='labels.json', type=str)
    parser.add_argument("--config_file", default='config/config.json', type=str)

    parser.add_argument("--do_train", action='store_true', default=True)  # False
    parser.add_argument("--do_eval", action='store_true', default=False)  # True
    parser.add_argument("--model_checkpoint", default=r'checkpoint/pair/re/nyt/finetuned|uie|epoch=0001|step=0000626.ckpt', type=str)
    parser.add_argument("--load_from_checkpoint", action='store_true', default=False)  # False
    parser.add_argument("--exp_version", default=2, type=int)

    parser.add_argument("--max_input_seq_length", default=256, type=int)
    parser.add_argument("--max_output_seq_length", default=156, type=int)

    parser.add_argument("--num_train_epochs", default=1000, type=int)
    parser.add_argument("--train_patience", default=100, type=int)
    parser.add_argument("--train_batch_size", default=40, type=int)
    parser.add_argument("--val_batch_size", default=40, type=int)

    parser.add_argument("--n_gpu", default=[0])
    parser.add_argument("--seed", default=42, type=int)

    args = parser.parse_args()

    args.device = 'cuda:' + str(args.n_gpu[0])

    args.data_path = f"data/{args.task_type}/{args.task}/{args.data}"
    if not os.path.exists(args.data_path):
        raise FileNotFoundError('no such data path')

    args.log_path = f"{args.task_type}/{args.task}/{args.data}"
    args.log_sub_path = f"logs/{args.task_type}/{args.task}/{args.data}/version_{args.exp_version}"
    if not os.path.exists(args.log_sub_path):
        os.makedirs(args.log_sub_path, exist_ok=True)

    args.output_dir = f"checkpoint/{args.task_type}/{args.task}/{args.data}"
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    return args


def make_dataloader(tokenizer_location, data_path, task, task_type, train_file, val_file, test_file, label_file, max_input_seq_length,
                    max_output_seq_length, train_batch_size, val_batch_size, mylogger):
    train_data, dev_data, test_data = loading_data(tokenizer_location, data_path, task, task_type, train_file, val_file, test_file, label_file,
                                                   max_input_seq_length, max_output_seq_length, logger=mylogger, is_save=True)

    if dev_data != None:
        val_dataset = MyDataset(dev_data)
        val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, collate_fn=val_dataset.collate_fn,
                                    shuffle=False, num_workers=8)
    else:
        val_dataloader = None

    if train_data != None:
        train_dataset = MyDataset(train_data)
        train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, collate_fn=train_dataset.collate_fn,
                                      shuffle=True, num_workers=8)
    else:
        train_dataloader = None

    return val_dataloader, train_dataloader


if __name__ == '__main__':
    args = init_args()
    with open(args.config_file, encoding='utf-8') as f:
        config = json.load(f)

    mylogger = get_logger(os.path.join(args.log_sub_path, 'output_finetune.log'))
    mylogger.info('\n' * 5)
    mylogger.info('==' * 20 + time.asctime(time.localtime()) + '==' * 20)
    mylogger.info('**' * 5 + 'args' + '**' * 5)
    mylogger.info(args)
    mylogger.info('**' * 5 + 'config' + '**' * 5)
    mylogger.info(config)

    seed_everything(args.seed)

    if args.do_train:
        val_dataloader, train_dataloader = make_dataloader(config["lm_location"], args.data_path, args.task, args.task_type,
                                                           args.train_file, args.eval_file, None, args.label_file,
                                                           args.max_input_seq_length, args.max_output_seq_length,
                                                           args.train_batch_size, args.val_batch_size, mylogger)

        if args.load_from_checkpoint:
            model_wrapper = ModelWrapper.load_from_checkpoint(args.model_checkpoint)
        else:
            model_wrapper = ModelWrapper(**config)

        cb_checkpoint = ModelCheckpoint(
            dirpath=args.output_dir, filename='finetuned|%s|{epoch:04d}|{step:07d}' % (config["model_type"]),
            monitor='rouge', mode='max', save_top_k=1
        )
        cb_earlystop = EarlyStopping(monitor="rouge", patience=args.train_patience, mode="max")
        logger = TensorBoardLogger('logs', name=args.log_path, version=args.exp_version)

        train_params = dict(
            default_root_dir=args.output_dir,
            accumulate_grad_batches=config["gradient_accumulation_steps"],
            devices=args.n_gpu,
            accelerator="gpu",
            strategy="dp",
            gradient_clip_val=1.0,
            max_epochs=args.num_train_epochs,
            callbacks=[cb_checkpoint, cb_earlystop],
            logger=logger
        )

        mylogger.info("**" * 15 + "Start Fine-tuning" + "**" * 15)
        trainer = Trainer(**train_params)
        trainer.fit(model_wrapper, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


    elif args.do_eval:
        val_dataloader, _ = make_dataloader(config["lm_location"], args.data_path, args.task, args.task_type,
                                            None, args.eval_file, None, args.label_file,
                                            args.max_input_seq_length, args.max_output_seq_length,
                                            args.train_batch_size, args.val_batch_size, mylogger)

        model_wrapper = ModelWrapper.load_from_checkpoint(args.model_checkpoint)
        model_wrapper.is_train = False
        model_wrapper.to(args.device)

        logger = TensorBoardLogger('logs', name=args.log_path, version=args.exp_version)
        val_params = dict(
            devices=args.n_gpu,
            accelerator="gpu",
            logger=logger
        )

        mylogger.info("**" * 15 + "Start Evaluating" + "**" * 15)
        trainer = Trainer(**val_params)
        val_metric = trainer.validate(model_wrapper, val_dataloader)
