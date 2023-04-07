#!/usr/bin/env python
# coding=utf-8

task_types = ['span', 'pair', 'hyperpair']


class TaskType:
    SPAN = 'span'
    PAIR = 'pair'
    HYPERPAIR = 'hyperpair'


class ModelType:
    UIE = 'uie'
    LASUIE = 'lasuie'
    LASUIE_STRUCT_TUNING = 'lasuie-struct-tuning'


class TunedType:
    FINETUNED = 'finetuned'
    STRUCT_FINETUNED = 'struct_finetuned'


class LHESequenceMarker:
    seq_start = '<extra_id_0>'
    seq_end = '<extra_id_1>'
    sep = '<extra_id_2>'
    span_start = '<extra_id_0>'
    span_end = '<extra_id_1>'
    sub_span_start = '<extra_id_3>'
    rel_start = '<extra_id_0>'
    null = '<extra_id_4>'


class ReadableMarker:
    left_bracket = '【'
    right_bracket = '】'
    brackets = left_bracket + right_bracket
    sep_marker = '，'
    rel_end_marker = '￥'
    null_marker = '[NULL]'
