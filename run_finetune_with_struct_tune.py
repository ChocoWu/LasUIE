#!/usr/bin/env python
# coding=utf-8

import os
import argparse
import time
import json
import numpy as np
from torchsummary import summary
from tqdm import tqdm
import gc

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything, Trainer, LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import get_linear_schedule_with_warmup, AdamW, T5Tokenizer
from transformers import LogitsProcessorList, MinLengthLogitsProcessor, BeamSearchScorer, \
    NoRepeatNGramLogitsProcessor, ForcedEOSTokenLogitsProcessor, \
    ForcedBOSTokenLogitsProcessor, StoppingCriteriaList, MaxLengthCriteria

from engine.module import StrcutT5ForConditionalGeneration, StructFinetuner, \
    StrcutFTT5ForConditionalGeneration, T5ForConditionalGeneration
from engine.data_utils import MyDataset, loading_data
from engine.evaluating import measuring, Evaluation
from engine.constants import ModelType, TaskType
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
    parser.add_argument("--config_file", default='config/config_struct_tune.json', type=str)

    parser.add_argument("--load_from_checkpoint", action='store_true', help="much load from warm-uply tuned model", default=True)  # False
    parser.add_argument("--model_checkpoint", default=r'checkpoint/pair/re/nyt/finetuned|lasuie|epoch=0000|step=0000313-v1.ckpt', type=str)
    parser.add_argument("--exp_version", default=1, type=int)

    parser.add_argument("--max_input_seq_length", default=256, type=int)
    parser.add_argument("--max_output_seq_length", default=156, type=int)

    parser.add_argument("--struct_tune_iterations", default=50, type=int)
    parser.add_argument("--num_beams", default=2, type=int)
    parser.add_argument("--rl_hidden_dim1", default=256, type=int)
    parser.add_argument("--rl_hidden_dim2", default=32, type=int)
    parser.add_argument("--rl_lr", default=5e-4, type=float)

    parser.add_argument("--num_train_epochs", default=1000, type=int)
    parser.add_argument("--train_patience", default=100, type=int)
    parser.add_argument("--train_batch_size", default=1, type=int)
    parser.add_argument("--val_batch_size", default=1, type=int)

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

    args.output_rl_dir = f"checkpoint/{args.task_type}/{args.task}/{args.data}/struct_tuning"
    if not os.path.exists(args.output_rl_dir):
        os.makedirs(args.output_rl_dir, exist_ok=True)

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


def struct_tuning(struct_finetuner, task_type, pretrain_model, tokenizer, train_dataloader, val_dataloader, max_input_seq_length, max_output_seq_length, search_nums, num_beams, device):
    def optimize_modal(model, reward):
        optimizer = AdamW(model.parameters(), lr=model.hparams.learning_rate)
        neg_log_prob = F.cross_entropy(input=encoder_input_ids, target=gold_outputs)

        loss = neg_log_prob * reward
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

    metric_up = []
    for train_batch in tqdm(train_dataloader, desc='instance',  leave=False, colour='red'):
        golds = tokenizer.batch_decode(train_batch['output_ids'], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        summary_ids = pretrain_model.model.generate(input_ids=train_batch['input_ids'].to(device),
                                                    attention_mask=train_batch['masks'].to(device))
        preds = tokenizer.batch_decode(summary_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

        preds = [postprocess_text(x, tokenizer) for x in preds]
        golds = [postprocess_text(x, tokenizer) for x in golds]

        scores = measuring([Evaluation.ROUGE], preds, golds, task_type)
        baseline_rouge_score = scores['rouge']

        batch_size = train_batch['input_ids'].size(0)
        encoder_input_ids = train_batch['input_ids'].repeat_interleave(num_beams, dim=0).to(device)
        encoder_attention_mask = train_batch['masks'].repeat_interleave(num_beams, dim=0).to(device)

        gold_outputs = train_batch['output_ids'].repeat_interleave(num_beams, dim=0).to(device)

        avg_metric_improv = []
        for _ in range(search_nums):

            encoder_outputs = pretrain_model.model.get_encoder()(input_ids=encoder_input_ids,
                                               attention_mask=encoder_attention_mask, return_dict=True)
            encoder_output = encoder_outputs[0]
            length = torch.sum(encoder_attention_mask, dim=1)
            length = length.to(device, dtype=torch.int)
            over_length = torch.ones_like(length, device=device) * max_input_seq_length
            length_invalid = over_length - 1
            length_valid = torch.where(length < over_length, length, length_invalid)
            sentence_rep = []
            for ix, i in enumerate(length_valid):
                sentence_rep.append(encoder_output[ix, i, :])
            sentence_rep = torch.stack(sentence_rep)

            height_delta, action_height, distance_delta, action_distance = struct_finetuner.choose_action(sentence_rep)

            struct_rep, _, _, _, _ = pretrain_model.model.struct_modeling(encoder_input_ids,
                                                              encoder_output,
                                                              encoder_attention_mask,
                                                              distance_delta, height_delta)

            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=num_beams,
                device=device,
                length_penalty=struct_finetuner.config.length_penalty,
                do_early_stopping=struct_finetuner.config.early_stopping,
                num_beam_hyps_to_keep=1
            )

            stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=max_output_seq_length)])

            input_ids = torch.ones((num_beams*batch_size, 1), device=device, dtype=torch.long)
            input_ids = input_ids * struct_finetuner.config.decoder_start_token_id
            model_kwargs = {
                "attention_mask": encoder_attention_mask,
                "encoder_outputs": encoder_outputs,
            }
            cur_play_outputs = pretrain_model.model.beam_search(input_ids.to(device),
                                                       beam_scorer=beam_scorer,
                                                       stopping_criteria=stopping_criteria,
                                                       pad_token_id=tokenizer.pad_token_id,
                                                       eos_token_id=tokenizer.eos_token_id,
                                                       output_scores=False,
                                                       past_key_values=struct_rep,
                                                       **model_kwargs)

            new_pred = tokenizer.batch_decode(cur_play_outputs, skip_special_tokens=False, clean_up_tokenization_spaces=False)

            new_pred = [postprocess_text(x, tokenizer) for x in new_pred]

            search_scores = measuring([Evaluation.ROUGE], new_pred, golds, task_type)
            search_rouge_score = search_scores['rouge']

            metric_improv = search_rouge_score - baseline_rouge_score
            if search_rouge_score != 0:
                reward = 0.9 * metric_improv / search_rouge_score
            else:
                reward = 0

            torch.cuda.empty_cache()

            struct_finetuner.learn(height_delta, distance_delta, action_height, action_distance)

            if baseline_rouge_score > search_rouge_score:
                avg_metric_improv.append(metric_improv)
                optimize_modal(pretrain_model, reward)

            else:
                break

            torch.cuda.empty_cache()

        avg_metric_improv = round(sum(avg_metric_improv)/len(avg_metric_improv), 4)
        metric_up.append(avg_metric_improv)

    return round(sum(metric_up)/len(metric_up), 4)


def load_checkpoint(model_wrapper, unstructed_model_wrapper):
    model_wrapper.model.load_state_dict(unstructed_model_wrapper.model.state_dict())
    return model_wrapper


if __name__ == '__main__':
    args = init_args()
    with open(args.config_file, encoding='utf-8') as f:
        config = json.load(f)

    mylogger = get_logger(os.path.join(args.log_sub_path, 'output_struct_tune.log'))
    mylogger.info('\n'*5)
    mylogger.info('=='*20 + time.asctime(time.localtime()) + '=='*20)
    mylogger.info('**'*5 + 'args' + '**'*5)
    mylogger.info(args)
    mylogger.info('**'*5 + 'config' + '**'*5)
    mylogger.info(config)

    seed_everything(args.seed)

    val_dataloader, train_dataloader = make_dataloader(config["lm_location"], args.data_path, args.task, args.task_type,
                                                       args.train_file, args.eval_file, None, args.label_file,
                                                       args.max_input_seq_length, args.max_output_seq_length,
                                                       args.train_batch_size, args.val_batch_size, mylogger)

    model_wrapper = ModelWrapper(**config)
    if args.load_from_checkpoint:
        unstructed_model_wrapper = ModelWrapper.load_from_checkpoint(args.model_checkpoint)
        model_wrapper = load_checkpoint(model_wrapper, unstructed_model_wrapper)
        del unstructed_model_wrapper
        gc.collect()
        torch.cuda.empty_cache()
    model_wrapper.to(args.device)

    struct_finetuner = StructFinetuner(config["lm_location"], seq_length=args.max_input_seq_length,
                                       input_dim=model_wrapper.model.config.d_model,
                                       hidden_dim1=args.rl_hidden_dim1, hidden_dim2=args.rl_hidden_dim2,
                                       output_dim=1, lr=args.rl_lr, device=args.device)

    mylogger.info("**" * 15 + "Start Structure Fine-tuning" + "**" * 15)

    val_params = dict(
        devices=args.n_gpu,
        accelerator="gpu",
        logger=TensorBoardLogger('logs', name=args.log_path)
    )

    trainer = Trainer(**val_params)

    max_metric = -np.inf
    for iter_ in tqdm(range(args.struct_tune_iterations), desc='epoch', colour='green'):
        metric_up = struct_tuning(struct_finetuner, args.task_type, model_wrapper, struct_finetuner.tokenizer, train_dataloader, val_dataloader,
                                  args.max_input_seq_length, args.max_output_seq_length, search_nums=10, num_beams=args.num_beams, device=args.device)

        mylogger.info(f'improved metric after structural finetuning: {metric_up}')

        save_file_struct_finetuner = os.path.join(args.output_rl_dir, f'struct-finetuner|{config["model_type"]}|step:{iter_}.pt')
        torch.save(struct_finetuner, save_file_struct_finetuner)

        val_metric = trainer.validate(model_wrapper, val_dataloader)
        val_score = val_metric[0]['rouge']
        if val_score > max_metric:
            max_metric = val_score
            save_file_model_wrapper = os.path.join(args.output_rl_dir, f'struct_tuned|{config["model_type"]}|step:{iter_}.ckpt')
            trainer.save_checkpoint(save_file_model_wrapper)
            mylogger.info(f'new model\' performance after structural finetuning: {max_metric}')
            mylogger.info(f'new model saved to {save_file_model_wrapper}')
