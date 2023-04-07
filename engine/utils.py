#!/usr/bin/env python
# coding=utf-8

import re
import logging
import sys
from nltk.tree import ParentedTree

from engine.constants import *

split_bracket = re.compile(r"<extra_id_\d>")


def get_logger(logfileName):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        # format="[%(asctime)s] [%(levelname)s] %(message)s",
        format="%(message)s",
        handlers=[
            logging.FileHandler(logfileName),
            logging.StreamHandler(sys.stdout)
        ]
    )
    mylogger = logging.getLogger(__name__)
    return mylogger


def clear_null(predictions, golds):
    predictions = [item if item != '' else ReadableMarker.null_marker for item in predictions]
    golds = [item if item != '' else ReadableMarker.null_marker for item in golds]
    return predictions, golds


def clean_text(tree_str):
    count = 0
    sum_count = 0

    tree_str_list = tree_str.split()

    for index, char in enumerate(tree_str_list):
        if char == ReadableMarker.left_bracket:
            count += 1
            sum_count += 1
        elif char == ReadableMarker.right_bracket:
            count -= 1
            sum_count += 1
        else:
            pass
        if count == 0 and sum_count > 0:
            return ' '.join(tree_str_list[:index + 1])
    return ' '.join(tree_str_list)


def add_space(text):
    new_text_list = list()
    for item in zip(split_bracket.findall(text), split_bracket.split(text)[1:]):
        new_text_list += item
    return ' '.join(new_text_list)


def check_is_null(text):
    tokens = text.split()
    if len(tokens) == 3:
        if ReadableMarker.null_marker == tokens[1].strip():
            return True


def convert_marker(text):
    text = add_space(text)
    for start in [LHESequenceMarker.span_start]:
        text = text.replace(start, ReadableMarker.left_bracket)

    for sep in [LHESequenceMarker.sep]:
        text = text.replace(sep, ReadableMarker.sep_marker)

    for sub_span_start in [LHESequenceMarker.sub_span_start]:
        text = text.replace(sub_span_start, ReadableMarker.rel_end_marker)

    for null in [LHESequenceMarker.null]:
        text = text.replace(null, ReadableMarker.null_marker)

    for end in [LHESequenceMarker.span_end]:
        text = text.replace(end, ReadableMarker.right_bracket)
    return text


def get_tree_str(tree):
    str_list = list()
    for element in tree:
        if isinstance(element, str):
            str_list += [element]
    return ' '.join(str_list)


def split_noisy_span_rel(label, span, sep_symbol=ReadableMarker.sep_marker,
                         sub_span_symbol=ReadableMarker.rel_end_marker):
    label_span = label + ' ' + span

    if (sub_span_symbol in label_span) and (sep_symbol in label_span):
        splited_rel_span_attr = re.split('%s|%s' % (sub_span_symbol, sep_symbol), label_span)
        splited_rel_span_attr = [item.strip() for item in splited_rel_span_attr if item.strip() != '']
        if len(splited_rel_span_attr) < 3:
            return splited_rel_span_attr[0], ReadableMarker.null_marker, splited_rel_span_attr[-1]
        elif len(splited_rel_span_attr) > 3:
            rel = splited_rel_span_attr[0]
            func_span = ReadableMarker.null_marker
            attr = splited_rel_span_attr[-1]
            for sps in splited_rel_span_attr[1:-1]:
                if sps != ReadableMarker.null_marker and sps != '': func_span = sps
            return rel, func_span, attr

        elif len(splited_rel_span_attr) == 3:
            return splited_rel_span_attr[0].strip(), splited_rel_span_attr[1].strip(), \
                splited_rel_span_attr[2].strip()

    elif sub_span_symbol in label_span and (sep_symbol not in label_span):
        splited_rel_span = re.split('%s' % (sub_span_symbol), label_span)
        splited_rel_span = [item.strip() for item in splited_rel_span if item.strip() != '']
        if len(splited_rel_span) < 2:
            return splited_rel_span[0], ReadableMarker.null_marker, ReadableMarker.null_marker
        elif len(splited_rel_span) > 2:
            rel = splited_rel_span[0]
            func_span = ReadableMarker.null_marker
            for sps in splited_rel_span[1:]:
                if sps != ReadableMarker.null_marker and sps != '': func_span = sps
            return rel, func_span, ReadableMarker.null_marker
        elif len(splited_rel_span) == 2:
            return splited_rel_span[0], splited_rel_span[1], ReadableMarker.null_marker

    # above two are for pair extract
    elif sep_symbol in label_span and (sub_span_symbol not in label_span):
        splited_span_attr = re.split('%s' % (sep_symbol), label_span)
        splited_span_attr = [item.strip() for item in splited_span_attr if item.strip() != '']
        if len(splited_span_attr) > 2:
            func_span = ReadableMarker.null_marker
            for sps in splited_span_attr[:-1]:
                if sps != ReadableMarker.null_marker and sps != '': func_span = sps
            return func_span, splited_span_attr[-1]
        elif len(splited_span_attr) == 2:
            return splited_span_attr[0], splited_span_attr[1]

    return label, span


def form_check(sequence, task_type):
    token_list = sequence.split()

    def check_bracket_balance(token_list, task_type):
        """
        check whether illed left_bracket right_bracket
        Count Bracket Number (num_left - num_right), 0 indicates num_left = num_right
        """
        if task_type == TaskType.SPAN:
            max_depth = 2
        else:
            max_depth = 3

        if token_list[0].strip() != ReadableMarker.left_bracket:
            token_list.insert(0, ReadableMarker.left_bracket)

        if token_list[-1].strip() != ReadableMarker.right_bracket:
            token_list.append(ReadableMarker.right_bracket)

        left_count = 0
        right_count = 0
        to_del_pos = []
        inner_count = 0
        for posi, token in enumerate(token_list):
            token = token.strip()
            if token == ReadableMarker.left_bracket:
                right_count -= 1
                left_count += 1
                if left_count > max_depth:
                    to_del_pos.append(posi)
                inner_count += 1

            elif token == ReadableMarker.right_bracket:
                left_count -= 1
                right_count += 1
                if left_count > max_depth:
                    to_del_pos.append(posi)
                inner_count -= 1

            else:
                pass

        for i in sorted(to_del_pos, reverse=True):
            del token_list[i]

        if left_count > right_count:
            token_list += [ReadableMarker.right_bracket] * inner_count

        return token_list

    token_list = check_bracket_balance(token_list, task_type)

    def check_span_attr_correctness(token_list):
        '''
        check whether span is correct or null or too many: just one span+attr
        check whether too many sep_marker
        '''

        to_assert_pos = []
        for posi, token in enumerate(token_list):
            token = token.strip()
            if token == ReadableMarker.sep_marker:
                if token_list[posi - 1].strip() in [ReadableMarker.left_bracket]:
                    to_assert_pos.append(posi)

        for i in sorted(to_assert_pos, reverse=True):
            token_list.insert(i, ReadableMarker.null_marker)

        return token_list

    token_list = check_span_attr_correctness(token_list)

    return ' '.join(token_list)


def extract_input_sent(lhe_input):
    raw_sent = lhe_input.split(LHESequenceMarker.sep)[1].strip()
    return raw_sent


def extract_uie_labels(content_list, task_type):
    uie_labels = []
    if task_type == TaskType.SPAN:
        for items in content_list:
            span = items['main-span']
            span_attr = items['main-attr']
            uie_labels.append({"span": span, "attr": span_attr})

    elif task_type == TaskType.PAIR:
        for trp in content_list:
            span_s = trp['main-span']
            span_s_attr = trp['main-attr']
            sub_structs = trp['sub-struct']
            rel = sub_structs[0]['sub-rel']
            span_e = sub_structs[0]['sub-span']
            span_e_attr = sub_structs[0]['sub-attr']

            uie = {"span-s": {"span": span_s, "attr": span_s_attr},
                   "rel": rel,
                   "span-e": {"span": span_e, "attr": span_e_attr}
                   }
            uie_labels.append(uie)

    elif task_type == TaskType.HYPERPAIR:
        for trp in content_list:
            span_s = trp['main-span']
            span_s_attr = trp['main-attr']
            sub_structs = trp['sub-struct']
            sub_spans = []
            for sub_stru in sub_structs:
                sub_rel = sub_stru['sub-rel']
                sub_span = sub_stru['sub-span']
                sub_span_sttr = sub_stru['sub-attr']
                sub_spans.append({
                    "rel": sub_rel, "span": sub_span, "attr": sub_span_sttr
                })
            uie = {"span-s": {"span": span_s, "attr": span_s_attr},
                   "span-e": sub_spans
                   }
            uie_labels.append(uie)
    return uie_labels


def decoding_labels(model_output, task_type):
    sentences = []
    lhes = []
    ie_labels = []

    for batch in model_output:
        sents_ = batch['sentences']
        for lhe_input in sents_:
            raw_sent = extract_input_sent(lhe_input)
            sentences.append(raw_sent)

        lhes_ = batch['labels']
        for lhe_output in lhes_:
            lhe_output = convert_marker(lhe_output)
            lhes.append(lhe_output.strip())
            if check_is_null(lhe_output):
                ie_labels.append('')
            else:
                lhe_output = clean_text(lhe_output)
                lhe_output = form_check(lhe_output, task_type)
                try:
                    lhe_tree = ParentedTree.fromstring(lhe_output, brackets=ReadableMarker.brackets)
                except:
                    lhe_tree = ''

                if lhe_tree == '':
                    ie_labels.append('')
                else:
                    triplet_label_list, record_list = get_uie_list(lhe_tree)
                    uie_labels = extract_uie_labels(record_list, task_type)
                    ie_labels.append(uie_labels)

    uie_results = []
    for sent_, lhe_, ie_lb_ in zip(sentences, lhes, ie_labels):
        uie_results.append(
            {
                "sentence": sent_,
                "lhe": lhe_,
                task_type + "s": ie_lb_,
            }
        )
    return uie_results


def get_uie_list(main_tree):
    """ Convert LHE expression to UIE structure records
    """

    triplet_list = list()
    record_list = list()

    # how many triggers and the triggerred associated sub-struct
    for start_tree in main_tree:
        # Drop incomplete tree
        if isinstance(start_tree, str) or len(start_tree) == 0:
            continue

        start_span_type = start_tree.label()
        start_span_text = get_tree_str(start_tree)

        splited = split_noisy_span_rel(start_span_type, start_span_text)
        if len(splited) == 2:
            start_span, start_attr = splited[0], splited[1]
        elif len(splited) == 3:
            start_rel, start_span, start_attr = splited[0], splited[1], splited[2]

        # Drop empty generated span
        if start_span is None or start_attr == LHESequenceMarker.null:
            continue
        # Drop empty generated type
        if start_attr is None:
            continue

        record = {
            'main-span': start_span,
            'main-attr': start_attr,
            'sub-struct': list(),
        }

        sub_triplets = []
        # sub-struct
        for sub_tree in start_tree:
            if isinstance(sub_tree, str) or len(sub_tree) < 1:
                continue

            sub_span_type = sub_tree.label()
            sub_span_text = get_tree_str(sub_tree)

            splited_sub_span_attr = split_noisy_span_rel(sub_span_type, sub_span_text)
            if len(splited_sub_span_attr) == 2:
                sub_span, sub_attr = splited_sub_span_attr[0], splited_sub_span_attr[1]
            elif len(splited_sub_span_attr) == 3:
                sub_rel, sub_span, sub_attr = splited_sub_span_attr[0], splited_sub_span_attr[1], splited_sub_span_attr[2]

            # Drop empty generated span
            if sub_span is None or sub_attr == LHESequenceMarker.null:
                continue
            # Drop empty generated type
            if sub_attr is None:
                continue

            sub_triplets.append(((start_span, start_attr), sub_rel, (sub_span, sub_attr)))
            record['sub-struct'] += [{
                'sub-rel': sub_rel,
                'sub-span': sub_span,
                'sub-attr': sub_attr,
            }]

        if len(sub_triplets) == 0:
            sub_triplets.append((start_span, start_attr))
        triplet_list.extend(sub_triplets)

        record_list += [record]

    return triplet_list, record_list


def make_LHE(task_type_contents, task_type, task, sentence, span_attribute_list, rel_type_list=None, need_target_LHE=True):
    '''linearized hierarchical expression'''
    input_seq_cache, output_seq_cache = [], []

    # input seq
    input_seq_cache.append(LHESequenceMarker.seq_start)
    input_seq_cache.append(task)
    input_seq_cache.append(LHESequenceMarker.sep)
    input_seq_cache.append(sentence)
    input_seq_cache.append(LHESequenceMarker.sep)
    for attr in span_attribute_list[:-1]:
        input_seq_cache.append(attr)
        input_seq_cache.append(LHESequenceMarker.sep)
    input_seq_cache.append(span_attribute_list[-1])
    if rel_type_list != None:
        input_seq_cache.append(LHESequenceMarker.sep)
        for attr in rel_type_list[:-1]:
            input_seq_cache.append(attr)
            input_seq_cache.append(LHESequenceMarker.sep)
        input_seq_cache.append(rel_type_list[-1])
    input_seq_cache.append(LHESequenceMarker.seq_end)

    input_seq = ' '.join(input_seq_cache)

    if not need_target_LHE:
        return input_seq, None

    # output seq
    if task_type == TaskType.SPAN:
        output_seq_cache.append(LHESequenceMarker.seq_start)
        if len(task_type_contents) == 0:
            output_seq_cache.append(LHESequenceMarker.null)
        else:
            for span in task_type_contents:
                output_seq_cache.append(LHESequenceMarker.span_start)
                output_seq_cache.append(span['span'])
                output_seq_cache.append(LHESequenceMarker.sep)
                output_seq_cache.append(span['attr'])
                output_seq_cache.append(LHESequenceMarker.span_end)
        output_seq_cache.append(LHESequenceMarker.seq_end)

    elif task_type == TaskType.PAIR:
        output_seq_cache.append(LHESequenceMarker.seq_start)
        if len(task_type_contents) == 0:
            output_seq_cache.append(LHESequenceMarker.null)
        else:
            for pair in task_type_contents:
                output_seq_cache.append(LHESequenceMarker.span_start)
                output_seq_cache.append(pair['span-s']['span'])
                output_seq_cache.append(LHESequenceMarker.sep)
                output_seq_cache.append(pair['span-s']['attr'])

                output_seq_cache.append(LHESequenceMarker.rel_start)
                output_seq_cache.append(pair['rel'])

                output_seq_cache.append(LHESequenceMarker.sub_span_start)
                output_seq_cache.append(pair['span-e']['span'])
                output_seq_cache.append(LHESequenceMarker.sep)
                output_seq_cache.append(pair['span-e']['attr'])
                output_seq_cache.append(LHESequenceMarker.span_end)

                output_seq_cache.append(LHESequenceMarker.span_end)
        output_seq_cache.append(LHESequenceMarker.seq_end)

    elif task_type == TaskType.HYPERPAIR:
        output_seq_cache.append(LHESequenceMarker.seq_start)
        if len(task_type_contents) == 0:
            output_seq_cache.append(LHESequenceMarker.null)
        else:
            for pair in task_type_contents:
                output_seq_cache.append(LHESequenceMarker.span_start)
                output_seq_cache.append(pair['span-s']['span'])
                output_seq_cache.append(LHESequenceMarker.sep)
                output_seq_cache.append(pair['span-s']['attr'])

                for e_spans in pair['span-e']:
                    output_seq_cache.append(LHESequenceMarker.rel_start)
                    output_seq_cache.append(e_spans['rel'])

                    output_seq_cache.append(LHESequenceMarker.sub_span_start)
                    output_seq_cache.append(e_spans['span'])
                    output_seq_cache.append(LHESequenceMarker.sep)
                    output_seq_cache.append(e_spans['attr'])
                    output_seq_cache.append(LHESequenceMarker.span_end)

                output_seq_cache.append(LHESequenceMarker.span_end)
        output_seq_cache.append(LHESequenceMarker.seq_end)

    else:
        raise TypeError('not valid task type')

    output_seq = ' '.join(output_seq_cache)

    return input_seq, output_seq
