#!/usr/bin/env python
# coding=utf-8

from engine.utils import *
from engine.constants import *
from nltk.tree import ParentedTree
import numpy as np
from rouge import Rouge


class Evaluation:
    ROUGE = 'rouge'
    EXACT_MATCH = 'exact-match'
    UIE = 'uie-structure'


class Metric:
    def __init__(self, need_span_attr=False, need_rel_type=False, task_type=TaskType.SPAN, case_sense=False):
        self.tp_span = 0.
        self.gold_span_num = 0.
        self.pred_span_num = 0.

        self.tp_triple = 0.
        self.gold_triple_num = 0.
        self.pred_triple_num = 0.

        self.need_span_attr = need_span_attr
        self.need_rel_type = need_rel_type
        self.case_sense = case_sense
        self.task_type = task_type

    def __repr__(self) -> str:
        return f"tp_span: {self.tp_span}," \
               f" gold_span: {self.gold_span_num}," \
               f" pred_span: {self.pred_span_num}," \
               f" tp_triple: {self.tp_triple}," \
               f" gold_triple: {self.gold_triple_num}," \
               f" pred_triple: {self.pred_triple_num}"

    @staticmethod
    def safe_div(a, b):
        if b == 0.:
            return 0.
        else:
            return a / b

    def compute_f1(self):
        # span
        p_span, r_span = self.safe_div(self.tp_span, self.pred_span_num), self.safe_div(self.tp_span, self.gold_span_num)
        span_prefix = 'span_'
        result = {
            span_prefix + 'P': p_span,
            span_prefix + 'R': p_span,
            span_prefix + 'F1': self.safe_div(2 * p_span * r_span, p_span + r_span),
        }

        if self.task_type == TaskType.PAIR or self.task_type == TaskType.HYPERPAIR:
            # triple
            p_triple, r_triple = self.safe_div(self.tp_triple, self.pred_triple_num), self.safe_div(self.tp_triple, self.gold_triple_num)
            triple_prefix = 'triplet_'
            triple_result = {
                triple_prefix + 'P': p_triple,
                triple_prefix + 'R': r_triple,
                triple_prefix + 'F1': self.safe_div(2 * p_triple * r_triple, p_triple + r_triple),
            }
            result.update(triple_result)

        return result

    def count_instance(self, gold_list, pred_list):
        if self.task_type == TaskType.SPAN:
            gold_span_list = []
            pred_span_list = []
            for gd, pd in zip(gold_list, pred_list):
                span_gd = gd[0] if self.case_sense else gd[0].lower()
                attr_gd = gd[1] if self.case_sense else gd[1].lower()
                span_pd = pd[0] if self.case_sense else pd[0].lower()
                attr_pd = pd[1] if self.case_sense else pd[1].lower()
                if self.need_span_attr:
                    gold_span_list.append(span_gd + '-' + attr_gd)
                    pred_span_list.append(span_pd + '-' + attr_pd)
                else:
                    gold_span_list.append(span_gd)
                    pred_span_list.append(span_pd)

            gold_span_list = set(gold_span_list)
            pred_span_list = set(pred_span_list)

            self.gold_span_num += len(gold_span_list)
            self.pred_span_num += len(pred_span_list)
            self.tp_span += len(gold_span_list & pred_span_list)

        else:
            gold_span_list = []
            pred_span_list = []
            for gd, pd in zip(gold_list, pred_list):
                span_s_gd = gd[0][0] if self.case_sense else gd[0][0].lower()
                attr_s_gd = gd[0][1] if self.case_sense else gd[0][1].lower()
                span_s_pd = pd[0][0] if self.case_sense else pd[0][0].lower()
                attr_s_pd = pd[0][1] if self.case_sense else pd[0][1].lower()
                span_e_gd = gd[2][0] if self.case_sense else gd[2][0].lower()
                attr_e_gd = gd[2][1] if self.case_sense else gd[2][1].lower()
                span_e_pd = pd[2][0] if self.case_sense else pd[2][0].lower()
                attr_e_pd = pd[2][1] if self.case_sense else pd[2][1].lower()

                if self.need_span_attr:
                    gold_span_list.append(span_s_gd + '-' + attr_s_gd)
                    pred_span_list.append(span_s_pd + '-' + attr_s_pd)
                    gold_span_list.append(span_e_gd + '-' + attr_e_gd)
                    pred_span_list.append(span_e_pd + '-' + attr_e_pd)
                else:
                    gold_span_list.append(span_s_gd)
                    pred_span_list.append(span_s_pd)
                    gold_span_list.append(span_e_gd)
                    pred_span_list.append(span_e_pd)

            gold_span_list = set(gold_span_list)
            pred_span_list = set(pred_span_list)

            self.gold_span_num += len(gold_span_list)
            self.pred_span_num += len(pred_span_list)
            self.tp_span += len(gold_span_list & pred_span_list)

            gold_triple_list = []
            pred_triple_list = []
            for gd, pd in zip(gold_list, pred_list):
                span_s_gd = gd[0][0] if self.case_sense else gd[0][0].lower()
                attr_s_gd = gd[0][1] if self.case_sense else gd[0][1].lower()
                span_s_pd = pd[0][0] if self.case_sense else pd[0][0].lower()
                attr_s_pd = pd[0][1] if self.case_sense else pd[0][1].lower()

                span_e_gd = gd[2][0] if self.case_sense else gd[2][0].lower()
                attr_e_gd = gd[2][1] if self.case_sense else gd[2][1].lower()
                span_e_pd = pd[2][0] if self.case_sense else pd[2][0].lower()
                attr_e_pd = pd[2][1] if self.case_sense else pd[2][1].lower()

                rel_gd = gd[1] if self.case_sense else gd[1].lower()
                rel_pd = pd[1] if self.case_sense else pd[1].lower()

                if self.need_span_attr:
                    if self.need_rel_type:
                        gold_triple_list.append(span_s_gd + '-' + attr_s_gd + '-' + rel_gd + '-' + span_e_gd + '-' + attr_e_gd)
                        pred_triple_list.append(span_s_pd + '-' + attr_s_pd + '-' + rel_pd + '-' + span_e_pd + '-' + attr_e_pd)
                    else:
                        gold_triple_list.append(span_s_gd + '-' + attr_s_gd + '-' + span_e_gd + '-' + attr_e_gd)
                        pred_triple_list.append(span_s_pd + '-' + attr_s_pd + '-' + span_e_pd + '-' + attr_e_pd)
                else:
                    if self.need_rel_type:
                        gold_triple_list.append(span_s_gd + '-' + rel_gd + '-' + span_e_gd)
                        pred_triple_list.append(span_s_pd + '-' + rel_pd + '-' + span_e_pd)
                    else:
                        gold_triple_list.append(span_s_gd + '-' + span_e_gd)
                        pred_triple_list.append(span_s_pd + '-' + span_e_pd)

            gold_triple_list = set(gold_triple_list)
            pred_triple_list = set(pred_triple_list)

            self.gold_triple_num += len(gold_triple_list)
            self.pred_triple_num += len(pred_triple_list)
            self.tp_triple += len(gold_triple_list & pred_triple_list)

    def count_batch_instance(self, batch_gold_list, batch_pred_list):
        for gold_list, pred_list in zip(batch_gold_list, batch_pred_list):
            self.count_instance(gold_list=gold_list, pred_list=pred_list)


def measuring(eval_metircs, predictions, golds, task_type, need_span_attr=True, need_rel_type=True):
    uie_metric = Metric(need_span_attr=need_span_attr, need_rel_type=need_rel_type, task_type=task_type)
    rouge = Rouge()

    predictions, golds = clear_null(predictions, golds)
    if Evaluation.UIE not in eval_metircs:
        result = {
            "rouge": rouge.get_scores(predictions, golds, ignore_empty=True)[0]['rouge-1']['f']
        }
        result = {k: round(v * 100, 2) for k, v in result.items()}
        prediction_lens = [len(pred.split()) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return result

    for gold, pred in zip(golds, predictions):

        gold = convert_marker(gold)
        pred = convert_marker(pred)

        if check_is_null(gold) or check_is_null(pred):
            continue

        gold = clean_text(gold)
        pred = clean_text(pred)

        pred_after = form_check(pred, task_type)

        try:
            gold_tree = ParentedTree.fromstring(gold, brackets=ReadableMarker.brackets)
        except:
            continue

        try:
            pred_tree = ParentedTree.fromstring(pred_after, brackets=ReadableMarker.brackets)
        except:
            continue

        gold_triplet_list, gold_record_list = get_uie_list(gold_tree)
        pred_triplet_list, pred_record_list = get_uie_list(pred_tree)

        uie_metric.count_instance(gold_triplet_list, pred_triplet_list)

    result = {
        "rouge": rouge.get_scores(predictions, golds, ignore_empty=True)[0]['rouge-1']['f']
    }
    new_result = uie_metric.compute_f1()
    result.update(new_result)
    result = {k: round(v * 100, 2) for k, v in result.items()}

    prediction_lens = [len(pred.split()) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return result
