#!/usr/bin/env python
# coding=utf-8

import os
import argparse
import time
import json
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything, Trainer, LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import T5Tokenizer

from engine.module import StrcutT5ForConditionalGeneration, StructFinetuner, \
    StrcutFTT5ForConditionalGeneration, T5ForConditionalGeneration
from engine.data_utils import MyDataset, loading_data
from engine.evaluating import measuring, Evaluation
from engine.constants import ModelType, TaskType, TunedType
from engine.utils import get_logger, decoding_labels

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

    def predict_step(self, batch, batch_idx):
        pred_ids = self.model.generate(input_ids=batch['input_ids'], attention_mask=batch['masks'])
        preds = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        preds = [self.postprocess_text(x) for x in preds]

        sentences = self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        sentences = [self.postprocess_text(x) for x in sentences]

        return {'sentences': sentences,
                'labels': preds}

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
            avg_rouge_score = np.stack([x['rouge'] for x in outputs]).mean()
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
            self.log('span-F1', avg_span_F1)
            if self.task_type == TaskType.SPAN:
                mylogger.info(f'rouge: {avg_rouge_score} || span-F1: {avg_span_F1}')
                return {'rouge': avg_rouge_score,
                        'span-F1': avg_span_F1,
                        }
            else:
                avg_triplet_F1 = np.stack([x['triplet-F1'] for x in outputs]).mean()
                avg_triplet_F1 = round(float(avg_triplet_F1), 4)
                mylogger.info(f'rouge: {avg_rouge_score} || span-F1: {avg_span_F1} || triplet-F1: {avg_triplet_F1}')
                self.log('triplet-F1', avg_span_F1)
                return {'rouge': avg_rouge_score,
                        'span-F1': avg_span_F1,
                        'triplet-F1': avg_triplet_F1,
                        }

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default='re', type=str)
    parser.add_argument("--task_type", default='pair', help="span, pair, hyperpair", type=str)
    parser.add_argument("--data", default='nyt', type=str)

    parser.add_argument("--do_test", action='store_true',
                        help="if yes then need testing label in test_file",
                        default=True)  # False, True
    parser.add_argument("--test_file", default='test.json', type=str)
    parser.add_argument("--label_file", default='labels.json', type=str)
    parser.add_argument("--config_file", default='config/config.json', type=str)

    parser.add_argument("--model_checkpoint", default=r'checkpoint/pair/re/nyt/finetuned|uie|epoch=0001|step=0000626.ckpt', type=str)
    parser.add_argument("--exp_version", default=1, type=int)

    parser.add_argument("--max_input_seq_length", default=256, type=int)
    parser.add_argument("--max_output_seq_length", default=500, type=int)

    parser.add_argument("--batch_size", default=150, type=int)

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

    model_name = args.model_checkpoint.split('/')[-1][:-5]
    args.test_output_dir = f"test_output/{args.task_type}/{args.task}/{args.data}/version_{args.exp_version}"
    if not os.path.exists(args.test_output_dir):
        os.makedirs(args.test_output_dir, exist_ok=True)
    args.test_save_file = os.path.join(args.test_output_dir, f'prediction_{model_name}.json')

    return args


def make_dataloader(tokenizer_location, data_path, task, task_type, test_file, label_file, max_input_seq_length,
                    max_output_seq_length, batch_size, mylogger, need_target_label=True):
    _, _, test_data = loading_data(tokenizer_location, data_path, task, task_type, None, None, test_file, label_file,
                                   max_input_seq_length, max_output_seq_length, logger=mylogger,
                                   is_save=True, need_target_LHE=need_target_label)

    test_dataset = MyDataset(test_data, need_label=need_target_label)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=test_dataset.collate_fn,
                                 shuffle=False, num_workers=8)

    return test_dataloader


def postprocess_text(x_str, tokenizer):
    to_remove_token_list = []

    def init_removal_token_list(to_remove_token_list, tokenizer):
        if tokenizer.unk_token:
            to_remove_token_list += [tokenizer.unk_token]
        if tokenizer.eos_token:
            to_remove_token_list += [tokenizer.eos_token]
        if tokenizer.pad_token:
            to_remove_token_list += [tokenizer.pad_token]
        return to_remove_token_list

    to_remove_token_list = init_removal_token_list(to_remove_token_list, tokenizer)

    for to_remove_token in to_remove_token_list:
        x_str = x_str.replace(to_remove_token, '')
    return x_str.strip()


def save_results(uie_results, save_file_path):
    with open(save_file_path, 'w', encoding='utf8') as f:
        json.dump(uie_results, f, ensure_ascii=True)


if __name__ == '__main__':
    args = init_args()
    with open(args.config_file, encoding='utf-8') as f:
        config = json.load(f)

    mylogger = get_logger(os.path.join(args.log_sub_path, 'output_test.log'))
    mylogger.info('\n' * 5)
    mylogger.info('==' * 20 + time.asctime(time.localtime()) + '==' * 20)
    mylogger.info('**' * 5 + 'args' + '**' * 5)
    mylogger.info(args)
    mylogger.info('**' * 5 + 'config' + '**' * 5)
    mylogger.info(config)

    seed_everything(args.seed)

    model_wrapper = ModelWrapper.load_from_checkpoint(args.model_checkpoint)
    model_wrapper.to(args.device)
    model_wrapper.eval()
    model_wrapper.is_train=False

    mylogger.info('**' * 5 + 'using model with tuned-typed: ' + TunedType.STRUCT_FINETUNED + '**' * 5)

    logger = TensorBoardLogger('logs', name=args.log_path, version=args.exp_version)
    val_params = dict(
        devices=args.n_gpu,
        accelerator="gpu",
        logger=logger
    )

    trainer = Trainer(**val_params)
    test_dataloader = make_dataloader(config["lm_location"], args.data_path, args.task, args.task_type,
                                      args.test_file, args.label_file,
                                      args.max_input_seq_length, args.max_output_seq_length,
                                      args.batch_size, mylogger, need_target_label=args.do_test)

    if args.do_test:
        mylogger.info("**" * 15 + "Start Testing" + "**" * 15)
        val_metric = trainer.validate(model_wrapper, test_dataloader)
        mylogger.info('testing performances:')
        mylogger.info(val_metric)

    mylogger.info("**" * 15 + "Start Inference" + "**" * 15)

    model_output = trainer.predict(model_wrapper, test_dataloader)
    uie_results = decoding_labels(model_output, args.task_type)
    save_results(uie_results, args.test_save_file)

    mylogger.info("**" * 10 + "End Inference, prediction saved at " + args.test_save_file + "**" * 10)
