#!/usr/bin/env python
# coding=utf-8

import os
import time
import argparse
import json
import numpy as np
import sys
from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm


from datasets import load_dataset
import flax
import jax
import jax.numpy as jnp
import optax
from flax import jax_utils, traverse_util
from flax.training import train_state
from flax.training.common_utils import get_metrics, onehot, shard
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    FLAX_MODEL_FOR_MASKED_LM_MAPPING,
    AutoTokenizer,
    BatchEncoding,
    FlaxT5ForConditionalGeneration,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    T5Config,
    is_tensorboard_available,
    set_seed,
)
from transformers.models.t5.modeling_flax_t5 import shift_tokens_right
from transformers.utils import get_full_repo_name
from engine.constants import ModelType, TaskType
from engine.module import StrcutT5ForConditionalGeneration, StrcutFTT5ForConditionalGeneration, T5ForConditionalGeneration
from engine.utils import get_logger
from engine.cus_argument import *


model_dict = {
    ModelType.UIE: T5ForConditionalGeneration,
    ModelType.LASUIE: StrcutT5ForConditionalGeneration,
    ModelType.LASUIE_STRUCT_TUNING: StrcutFTT5ForConditionalGeneration,
}



def compute_input_and_target_lengths(inputs_length, noise_density, mean_noise_span_length):
    """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2466>`__ .

    Training parameters to avoid padding with random_spans_noise_mask.
    When training a model with random_spans_noise_mask, we would like to set the other
    training hyperparmeters in a way that avoids padding.
    This function helps us compute these hyperparameters.
    We assume that each noise span in the input is replaced by extra_tokens_per_span_inputs sentinel tokens,
    and each non-noise span in the targets is replaced by extra_tokens_per_span_targets sentinel tokens.
    This function tells us the required number of tokens in the raw example (for split_tokens())
    as well as the length of the encoded targets. Note that this function assumes
    the inputs and targets will have EOS appended and includes that in the reported length.

    Args:
        inputs_length: an integer - desired length of the tokenized inputs sequence
        noise_density: a float
        mean_noise_span_length: a float
    Returns:
        tokens_length: length of original text in tokens
        targets_length: an integer - length in tokens of encoded targets sequence
    """

    def _tokens_length_to_inputs_length_targets_length(tokens_length):
        num_noise_tokens = int(round(tokens_length * noise_density))
        num_nonnoise_tokens = tokens_length - num_noise_tokens
        num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
        # inputs contain all nonnoise tokens, sentinels for all noise spans
        # and one EOS token.
        _input_length = num_nonnoise_tokens + num_noise_spans + 1
        _output_length = num_noise_tokens + num_noise_spans + 1
        return _input_length, _output_length

    tokens_length = inputs_length

    while _tokens_length_to_inputs_length_targets_length(tokens_length + 1)[0] <= inputs_length:
        tokens_length += 1

    inputs_length, targets_length = _tokens_length_to_inputs_length_targets_length(tokens_length)

    # minor hack to get the targets length to be equal to inputs length
    # which is more likely to have been set to a nice round number.
    if noise_density == 0.5 and targets_length > inputs_length:
        tokens_length -= 1
        targets_length -= 1
    return tokens_length, targets_length


@flax.struct.dataclass
class FlaxDataCollatorForT5MLM:
    """
    Data collator used for T5 span-masked language modeling.
    It is made sure that after masking the inputs are of length `data_args.max_seq_length` and targets are also of fixed length.
    For more information on how T5 span-masked language modeling works, one can take a look
    at the `official paper <https://arxiv.org/pdf/1910.10683.pdf>`__
    or the `official code for preprocessing <https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py>`__ .

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        noise_density (:obj:`float`):
            The probability with which to (randomly) mask tokens in the input.
        mean_noise_span_length (:obj:`float`):
            The average span length of the masked tokens.
        input_length (:obj:`int`):
            The expected input length after masking.
        target_length (:obj:`int`):
            The expected target length after masking.
        pad_token_id: (:obj:`int`):
            The pad token id of the model
        decoder_start_token_id: (:obj:`int):
            The decoder start token id of the model
    """

    tokenizer: PreTrainedTokenizerBase
    noise_density: float
    mean_noise_span_length: float
    input_length: int
    target_length: int
    pad_token_id: int
    decoder_start_token_id: int

    def __call__(self, examples: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:

        # convert list to dict and tensorize input
        batch = BatchEncoding(
            {k: np.array([examples[i][k] for i in range(len(examples))]) for k, v in examples[0].items()}
        )

        input_ids = batch["input_ids"]
        batch_size, expandend_input_length = input_ids.shape

        mask_indices = np.asarray([self.random_spans_noise_mask(expandend_input_length) for i in range(batch_size)])
        labels_mask = ~mask_indices

        input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
        labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))

        batch["input_ids"] = self.filter_input_ids(input_ids, input_ids_sentinel)
        batch["labels"] = self.filter_input_ids(input_ids, labels_sentinel)

        if batch["input_ids"].shape[-1] != self.input_length:
            raise ValueError(
                f"`input_ids` are incorrectly preprocessed. `input_ids` length is {batch['input_ids'].shape[-1]}, but"
                f" should be {self.target_length}."
            )

        if batch["labels"].shape[-1] != self.target_length:
            raise ValueError(
                f"`labels` are incorrectly preprocessed. `labels` length is {batch['labels'].shape[-1]}, but should be"
                f" {self.target_length}."
            )

        # to check that tokens are correctly preprocessed, one can run `self.tokenizer.batch_decode(input_ids)` and `self.tokenizer.batch_decode(labels)` here...
        batch["decoder_input_ids"] = shift_tokens_right(
            batch["labels"], self.pad_token_id, self.decoder_start_token_id
        )

        return batch

    def create_sentinel_ids(self, mask_indices):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices)
        sentinel_ids = np.where(sentinel_ids != 0, (len(self.tokenizer) - sentinel_ids), 0)
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        input_ids = input_ids_full[input_ids_full >= 0].reshape((batch_size, -1))
        input_ids = np.concatenate(
            [input_ids, np.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=np.int32)], axis=-1
        )
        return input_ids

    def random_spans_noise_mask(self, length):

        """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .

        Noise mask consisting of random spans of noise tokens.
        The number of noise tokens and the number of noise spans and non-noise spans
        are determined deterministically as follows:
        num_noise_tokens = round(length * noise_density)
        num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        Spans alternate between non-noise and noise, beginning with non-noise.
        Subject to the above restrictions, all masks are equally likely.

        Args:
            length: an int32 scalar (length of the incoming token sequence)
            noise_density: a float - approximate density of output mask
            mean_noise_span_length: a number

        Returns:
            a boolean tensor with shape [length]
        """

        orig_length = length

        num_noise_tokens = int(np.round(length * self.noise_density))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(np.round(num_noise_tokens / self.mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2]
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]


def generate_batch_splits(samples_idx: jnp.ndarray, batch_size: int) -> jnp.ndarray:
    num_samples = len(samples_idx)
    samples_to_remove = num_samples % batch_size

    if samples_to_remove != 0:
        samples_idx = samples_idx[:-samples_to_remove]
    sections_split = num_samples // batch_size
    batch_idx = np.split(samples_idx, sections_split)
    return batch_idx


def write_train_metric(summary_writer, train_metrics, train_time, step):
    summary_writer.scalar("train_time", train_time, step)

    train_metrics = get_metrics(train_metrics)
    for key, vals in train_metrics.items():
        tag = f"train_{key}"
        for i, val in enumerate(vals):
            summary_writer.scalar(tag, val, step - len(vals) + i + 1)


def write_eval_metric(summary_writer, eval_metrics, step):
    for metric_name, value in eval_metrics.items():
        summary_writer.scalar(f"eval_{metric_name}", value, step)



def get_model(model_type, lm_location):
    return model_dict[model_type].from_pretrained(lm_location)




def start_training(model_args, data_args, training_args, logger):

    if training_args.push_to_hub:
        if training_args.hub_model_id is None:
            repo_name = get_full_repo_name(
                Path(training_args.output_dir).absolute().name, token=training_args.hub_token
            )
        else:
            repo_name = training_args.hub_model_id
        repo = Repository(training_args.output_dir, clone_from=repo_name)

    if data_args.dataset_name is not None:
        datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        if "validation" not in datasets.keys():
            datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
            datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = data_args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        if "validation" not in datasets.keys():
            datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
            datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.config_name:
        config = T5Config.from_pretrained(
            model_args.config_name,
            cache_dir=model_args.cache_dir,
            vocab_size=len(tokenizer),
            use_auth_token=True if model_args.use_auth_token else None,
        )
    elif model_args.model_name_or_path:
        config = T5Config.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name], return_attention_mask=False)

    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    expanded_inputs_length, targets_length = compute_input_and_target_lengths(
        inputs_length=max_seq_length,
        noise_density=data_args.mlm_probability,
        mean_noise_span_length=data_args.mean_noise_span_length,
    )

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= expanded_inputs_length:
            total_length = (total_length // expanded_inputs_length) * expanded_inputs_length
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + expanded_inputs_length] for i in range(0, total_length, expanded_inputs_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    tokenized_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    # Enable tensorboard only on the master node
    has_tensorboard = is_tensorboard_available()
    if has_tensorboard and jax.process_index() == 0:
        try:
            from flax.metrics.tensorboard import SummaryWriter

            summary_writer = SummaryWriter(log_dir=Path(training_args.output_dir))
        except ImportError as ie:
            has_tensorboard = False
            logger.warning(
                f"Unable to display metrics through TensorBoard because some package are not installed: {ie}"
            )
    else:
        logger.warning(
            "Unable to display metrics through TensorBoard because the package is not installed: "
            "Please run pip install tensorboard to enable."
        )

    # Initialize our training
    rng = jax.random.PRNGKey(training_args.seed)
    dropout_rngs = jax.random.split(rng, jax.local_device_count())


    def initial_PT_model():
        if model_args.model_name_or_path:
            model = StrcutT5ForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                seed=training_args.seed,
                dtype=getattr(jnp, model_args.dtype),
                use_auth_token=True if model_args.use_auth_token else None,
            )
            model.set_post_training()
        else:
            config.vocab_size = len(tokenizer)
            model = FlaxT5ForConditionalGeneration(
                config,
                seed=training_args.seed,
                dtype=getattr(jnp, model_args.dtype),
                use_auth_token=True if model_args.use_auth_token else None,
            )

        return model

    model = initial_PT_model()

    data_collator = FlaxDataCollatorForT5MLM(
        tokenizer=tokenizer,
        noise_density=data_args.mlm_probability,
        mean_noise_span_length=data_args.mean_noise_span_length,
        input_length=max_seq_length,
        target_length=targets_length,
        pad_token_id=model.config.pad_token_id,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    num_epochs = int(training_args.num_train_epochs)
    train_batch_size = int(training_args.per_device_train_batch_size) * jax.device_count()
    eval_batch_size = int(training_args.per_device_eval_batch_size) * jax.device_count()

    num_train_steps = len(tokenized_datasets["train"]) // train_batch_size * num_epochs

    num_of_hosts = jax.process_count()
    current_host_idx = jax.process_index()

    # Create learning rate schedule
    warmup_fn = optax.linear_schedule(
        init_value=0.0, end_value=training_args.learning_rate, transition_steps=training_args.warmup_steps
    )
    decay_fn = optax.linear_schedule(
        init_value=training_args.learning_rate,
        end_value=0,
        transition_steps=num_train_steps - training_args.warmup_steps,
    )
    linear_decay_lr_schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, decay_fn], boundaries=[training_args.warmup_steps]
    )

    def decay_mask_fn(params):
        flat_params = traverse_util.flatten_dict(params)
        flat_mask = {
            path: (path[-1] != "bias" and path[-2:] not in [("layer_norm", "scale"), ("final_layer_norm", "scale")])
            for path in flat_params
        }
        return traverse_util.unflatten_dict(flat_mask)

    # create adam optimizer
    if training_args.adafactor:
        optimizer = optax.adafactor(
            learning_rate=linear_decay_lr_schedule_fn,
        )
    else:
        optimizer = optax.adamw(
            learning_rate=linear_decay_lr_schedule_fn,
            b1=training_args.adam_beta1,
            b2=training_args.adam_beta2,
            weight_decay=training_args.weight_decay,
            mask=decay_mask_fn,
        )

    # Setup train state
    state = train_state.TrainState.create(apply_fn=model.__call__, params=model.params, tx=optimizer)

    # Define gradient update step fn
    def train_step(state, batch, dropout_rng):
        dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

        def loss_fn(params):
            labels = batch.pop("labels")

            logits = state.apply_fn(**batch, params=params, dropout_rng=dropout_rng, train=True)[0]

            # compute loss
            loss = optax.softmax_cross_entropy(logits, onehot(labels, logits.shape[-1])).mean()

            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grad = grad_fn(state.params)
        grad = jax.lax.pmean(grad, "batch")
        new_state = state.apply_gradients(grads=grad)

        metrics = jax.lax.pmean(
            {"loss": loss, "learning_rate": linear_decay_lr_schedule_fn(state.step)}, axis_name="batch"
        )

        return new_state, metrics, new_dropout_rng

    p_train_step = jax.pmap(train_step, "batch", donate_argnums=(0,))

    def eval_step(params, batch):
        labels = batch.pop("labels")

        logits = model(**batch, params=params, train=False)[0]

        loss = optax.softmax_cross_entropy(logits, onehot(labels, logits.shape[-1]))

        accuracy = jnp.equal(jnp.argmax(logits, axis=-1), labels)

        metrics = {"loss": loss.mean(), "accuracy": accuracy.mean()}
        metrics = jax.lax.pmean(metrics, axis_name="batch")

        return metrics

    p_eval_step = jax.pmap(eval_step, "batch", donate_argnums=(0,))

    state = jax_utils.replicate(state)

    train_time = 0
    epochs = tqdm(range(num_epochs), desc="Epoch ... ", position=0)
    for epoch in epochs:
        # ======================== Training ================================
        train_start = time.time()
        train_metrics = []

        rng, input_rng = jax.random.split(rng)

        num_train_samples = len(tokenized_datasets["train"])
        train_samples_idx = np.random.permutation(np.arange(num_train_samples))
        train_batch_idx = generate_batch_splits(train_samples_idx, train_batch_size)

        for step, batch_idx in enumerate(tqdm(train_batch_idx, desc="Training...", position=1)):
            samples = [tokenized_datasets["train"][int(idx)] for idx in batch_idx]
            model_inputs = data_collator(samples)

            local_host_model_inputs = {
                key: np.split(model_inputs.data[key], num_of_hosts, axis=0)[current_host_idx]
                for key, value in model_inputs.data.items()
            }

            model_inputs = shard(local_host_model_inputs)
            state, train_metric, dropout_rngs = p_train_step(state, model_inputs, dropout_rngs)
            train_metrics.append(train_metric)

            cur_step = epoch * (num_train_samples // train_batch_size) + step

            if cur_step % training_args.logging_steps == 0 and cur_step > 0:
                # Save metrics
                train_metric = jax_utils.unreplicate(train_metric)
                train_time += time.time() - train_start
                if has_tensorboard and jax.process_index() == 0:
                    write_train_metric(summary_writer, train_metrics, train_time, cur_step)

                epochs.write(
                    f"Step... ({cur_step} | Loss: {train_metric['loss'].mean()}, Learning Rate:"
                    f" {train_metric['learning_rate'].mean()})"
                )

                train_metrics = []

            if cur_step % training_args.eval_steps == 0 and cur_step > 0:
                # ======================== Evaluating ==============================
                num_eval_samples = len(tokenized_datasets["validation"])
                eval_samples_idx = jnp.arange(num_eval_samples)
                eval_batch_idx = generate_batch_splits(eval_samples_idx, eval_batch_size)

                eval_metrics = []
                for i, batch_idx in enumerate(tqdm(eval_batch_idx, desc="Evaluating ...", position=2)):
                    samples = [tokenized_datasets["validation"][int(idx)] for idx in batch_idx]
                    model_inputs = data_collator(samples)

                    # Model forward
                    model_inputs = shard(model_inputs.data)
                    metrics = p_eval_step(state.params, model_inputs)
                    eval_metrics.append(metrics)

                # get eval metrics
                eval_metrics = get_metrics(eval_metrics)
                eval_metrics = jax.tree_map(jnp.mean, eval_metrics)

                # Update progress bar
                epochs.write(f"Step... ({cur_step} | Loss: {eval_metrics['loss']}, Acc: {eval_metrics['accuracy']})")

                # Save metrics
                if has_tensorboard and jax.process_index() == 0:
                    write_eval_metric(summary_writer, eval_metrics, cur_step)

            if cur_step % training_args.save_steps == 0 and cur_step > 0:
                # save checkpoint after each epoch and push checkpoint to the hub
                if jax.process_index() == 0:
                    params = jax.device_get(jax.tree_map(lambda x: x[0], state.params))
                    model.save_pretrained(training_args.output_dir, params=params)
                    tokenizer.save_pretrained(training_args.output_dir)
                    # if training_args.push_to_hub:
                    #     repo.push_to_hub(commit_message=f"Saving weights and logs of step {cur_step}", blocking=False)

    if training_args.do_eval:
        num_eval_samples = len(tokenized_datasets["validation"])
        eval_samples_idx = jnp.arange(num_eval_samples)
        eval_batch_idx = generate_batch_splits(eval_samples_idx, eval_batch_size)

        eval_metrics = []
        for i, batch_idx in enumerate(tqdm(eval_batch_idx, desc="Evaluating ...", position=2)):
            samples = [tokenized_datasets["validation"][int(idx)] for idx in batch_idx]
            model_inputs = data_collator(samples)

            # Model forward
            model_inputs = shard(model_inputs.data)
            metrics = p_eval_step(state.params, model_inputs)
            eval_metrics.append(metrics)

        # get eval metrics
        eval_metrics = get_metrics(eval_metrics)
        eval_metrics = jax.tree_map(lambda metric: jnp.mean(metric).item(), eval_metrics)

        if jax.process_index() == 0:
            eval_metrics = {f"eval_{metric_name}": value for metric_name, value in eval_metrics.items()}
            path = os.path.join(training_args.output_dir, "eval_results.json")
            with open(path, "w") as f:
                json.dump(eval_metrics, f, indent=4, sort_keys=True)


def init_args():

    parser = HfArgumentParser((PTModelArguments, PTDataTrainingArguments, PTTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir, exist_ok=True)

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )


    if not os.path.exists(training_args.log_dir):
        os.makedirs(training_args.log_dir, exist_ok=True)

    return model_args, data_args, training_args



if __name__ == '__main__':
    model_args, data_args, training_args = init_args()

    mylogger = get_logger(os.path.join(training_args.log_dir, 'output_post_training.log'))
    mylogger.info('\n'*5)
    mylogger.info('=='*20 + time.asctime(time.localtime()) + '=='*20)
    mylogger.info('**'*5 + 'Training/evaluation parameters' + '**'*5)
    mylogger.info(training_args)
    mylogger.info('**'*5 + 'Model parameters' + '**'*5)
    mylogger.info(model_args)
    mylogger.info('**'*5 + 'Data parameters' + '**'*5)
    mylogger.info(data_args)

    set_seed(training_args.seed)

    mylogger.info("**" * 15 + "Start Structure-aware Post-training" + "**" * 15)

    start_training(model_args, data_args, training_args, mylogger)
