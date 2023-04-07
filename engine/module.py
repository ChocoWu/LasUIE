#!/usr/bin/env python
# coding=utf-8

import dgl
import os
import pickle
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from transformers.models.t5.modeling_t5 import T5Model, logger, Seq2SeqLMOutput, BaseModelOutput, CrossEntropyLoss
from transformers import T5PreTrainedModel, T5Config, T5Tokenizer
from transformers.utils.model_parallel_utils import get_device_map

torch.manual_seed(42)


def cumprod(x, reverse=False, exclusive=False):
    """cumulative product."""
    if reverse:
        x = x.flip([-1])

    if exclusive:
        x = F.pad(x[:, :, :-1], (1, 0), value=1)

    cx = x.cumprod(-1)

    if reverse:
        cx = cx.flip([-1])
    return cx


def cumsum(x, reverse=False, exclusive=False):
    """cumulative sum."""
    bsz, _, length = x.size()
    device = x.device
    if reverse:
        if exclusive:
            w = torch.ones([bsz, length, length], device=device).tril(-1)
        else:
            w = torch.ones([bsz, length, length], device=device).tril(0)
        cx = torch.bmm(x, w)
    else:
        if exclusive:
            w = torch.ones([bsz, length, length], device=device).triu(1)
        else:
            w = torch.ones([bsz, length, length], device=device).triu(0)
        cx = torch.bmm(x, w)
    return cx


def cummin(x, reverse=False, exclusive=False, max_value=1e9):
    """cumulative min."""
    if reverse:
        if exclusive:
            x = F.pad(x[:, :, 1:], (0, 1), value=max_value)
        x = x.flip([-1]).cummin(-1)[0].flip([-1])
    else:
        if exclusive:
            x = F.pad(x[:, :, :-1], (1, 0), value=max_value)
        x = x.cummin(-1)[0]
    return x


def _get_activation_fn(activation):
    """Get specified activation function."""
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "leakyrelu":
        return nn.LeakyReLU()

    raise RuntimeError(
        "activation should be relu/gelu, not {}".format(activation))


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, h):
        z = self.fc(h)
        g.ndata['z'] = z
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)
        return g.ndata.pop('h')


class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(in_dim, out_dim))
        self.merge = merge

    def forward(self, g, h):
        head_out = [attn_head(g, h) for attn_head in self.heads]
        if self.merge == 'cat':
            return torch.cat(head_out, dim=1)
        else:
            return torch.mean(torch.stack(head_out))


class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(in_dim, hidden_dim, num_heads)
        self.layer2 = MultiHeadGATLayer(hidden_dim * num_heads, out_dim, 1)

    def forward(self, g, h):
        h = self.layer1(g, h)
        h = F.dropout(h, p=0.1, training=self.training)
        h = F.elu(h)
        h = F.dropout(h, p=0.1, training=self.training)
        h = self.layer2(g, h)
        h = F.dropout(h, p=0.1, training=self.training)
        return h


class T5ForConditionalGeneration(T5PreTrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
        r"decoder.embed_tokens.weight",
        r"lm_head.weight",
        r"final_logits_bias",
    ]

    _keys_to_ignore_on_load_unexpected = [
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.model_dim = config.d_model
        self.model = T5Model(config)

        self.config = config

        self.model.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.model.init_weights()

    def get_device_map(self):
        self.device_map = (get_device_map(len(self.model.encoder.block), range(torch.cuda.device_count())))
        return self.device_map

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.BoolTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            decoder_head_mask: Optional[torch.FloatTensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:
        """

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask

        # Encode
        if encoder_outputs is None:
            encoder = self.get_encoder()
            encoder_outputs = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = self.model._shift_right(labels)

        # Decode
        decoder = self.get_decoder()
        decoder_outputs = decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self.model._shift_right(labels)

    def _reorder_cache(self, past_key_values, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past_key_values is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past_key_values

        reordered_decoder_past = ()
        for layer_past_states in past_key_values:
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past


class Conv1d(nn.Module):

    def __init__(self, hidden_size, kernel_size, dilation=1):
        """Initialization.
        Args:
          hidden_size: dimension of input embeddings
          kernel_size: convolution kernel size
          dilation: the spacing between the kernel points
        """
        super(Conv1d, self).__init__()

        if kernel_size % 2 == 0:
            padding = (kernel_size // 2) * dilation
            self.shift = True
        else:
            padding = ((kernel_size - 1) // 2) * dilation
            self.shift = False
        self.conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size,
            padding=padding,
            dilation=dilation)

    def forward(self, x):
        """Compute convolution.
        Args:
          x: input embeddings
        Returns:
          conv_output: convolution results
        """

        if self.shift:
            return self.conv(x.transpose(1, 2)).transpose(1, 2)[:, 1:]
        else:
            return self.conv(x.transpose(1, 2)).transpose(1, 2)


class MultiheadAttention(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0.,
                 bias=True,
                 v_proj=True,
                 out_proj=True,
                 relative_bias=True):
        """Initialization.
        Args:
          embed_dim: dimension of input embeddings
          num_heads: number of self-attention heads
          dropout: dropout rate
          bias: bool, indicate whether include bias for linear transformations
          v_proj: bool, indicate whether project inputs to new values
          out_proj: bool, indicate whether project outputs to new values
          relative_bias: bool, indicate whether use a relative position based
            attention bias
        """

        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.drop = nn.Dropout(dropout)
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, ("embed_dim must be "
                                                             "divisible by "
                                                             "num_heads")

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        if v_proj:
            self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        else:
            self.v_proj = nn.Identity()

        if out_proj:
            self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        else:
            self.out_proj = nn.Identity()

        if relative_bias:
            self.relative_bias = nn.Parameter(torch.zeros((self.num_heads, 512)))
        else:
            self.relative_bias = None

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize attention parameters."""

        init.xavier_uniform_(self.q_proj.weight)
        init.constant_(self.q_proj.bias, 0.)

        init.xavier_uniform_(self.k_proj.weight)
        init.constant_(self.k_proj.bias, 0.)

        if isinstance(self.v_proj, nn.Linear):
            init.xavier_uniform_(self.v_proj.weight)
            init.constant_(self.v_proj.bias, 0.)

        if isinstance(self.out_proj, nn.Linear):
            init.xavier_uniform_(self.out_proj.weight)
            init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key_padding_mask=None, attn_mask=None):
        """Compute multi-head self-attention.
        Args:
          query: input embeddings
          key_padding_mask: 3D mask that prevents attention to certain positions
          attn_mask: 3D mask that rescale the attention weight at each position
        Returns:
          attn_output: self-attention output
        """

        length, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim

        head_dim = embed_dim // self.num_heads
        assert head_dim * self.num_heads == embed_dim, ("embed_dim must be "
                                                        "divisible by num_heads")
        scaling = float(head_dim) ** -0.5

        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)

        q = q * scaling

        if attn_mask is not None:
            assert list(attn_mask.size()) == [bsz * self.num_heads,
                                              query.size(0), query.size(0)]

        q = q.contiguous().view(length, bsz * self.num_heads,
                                head_dim).transpose(0, 1)
        k = k.contiguous().view(length, bsz * self.num_heads,
                                head_dim).transpose(0, 1)
        v = v.contiguous().view(length, bsz * self.num_heads,
                                head_dim).transpose(0, 1)

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(
            attn_output_weights.size()) == [bsz * self.num_heads, length, length]

        if self.relative_bias is not None:
            pos = torch.arange(length, device=query.device)
            relative_pos = torch.abs(pos[:, None] - pos[None, :]) + 256
            relative_pos = relative_pos[None, :, :].expand(bsz * self.num_heads, -1,
                                                           -1)

            relative_bias = self.relative_bias.repeat_interleave(bsz, dim=0)
            relative_bias = relative_bias[:, None, :].expand(-1, length, -1)
            relative_bias = torch.gather(relative_bias, 2, relative_pos)
            attn_output_weights = attn_output_weights + relative_bias

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights + key_padding_mask

        if attn_mask is None:
            attn_output_weights = torch.softmax(attn_output_weights, dim=-1)
        else:
            attn_output_weights = torch.sigmoid(attn_output_weights) * attn_mask

        attn_output_weights = self.drop(attn_output_weights)

        attn_output = torch.bmm(attn_output_weights, v)

        assert list(attn_output.size()) == [bsz * self.num_heads, length, head_dim]
        attn_output = attn_output.transpose(0, 1).contiguous().view(
            length, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output


class TransformerEncoder(nn.Module):

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 dropatt=0.1,
                 activation="leakyrelu",
                 relative_bias=True):
        """Initialization.
        Args:
          d_model: dimension of inputs
          nhead: number of self-attention heads
          dim_feedforward: dimension of hidden layer in feedforward layer
          dropout: dropout rate
          dropatt: drop attention rate
          activation: activation function
          relative_bias: bool, indicate whether use a relative position based
            attention bias
        """

        super(TransformerEncoder, self).__init__()
        self.self_attn = MultiheadAttention(
            d_model, nhead, dropout=dropatt, relative_bias=relative_bias)
        # Implementation of Feedforward model
        self.feedforward = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, dim_feedforward),
            _get_activation_fn(activation), nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model))

        self.norm = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.nhead = nhead

    def forward(self, src, attn_mask=None, key_padding_mask=None):
        """Pass the input through the encoder layer.

        Args:
          src: the sequence to the encoder layer (required).
          attn_mask: the mask for the src sequence (optional).
          key_padding_mask: the mask for the src keys per batch (optional).
        Returns:
          src3: the output of transformer layer, share the same shape as src.
        """
        src2 = self.self_attn(
            self.norm(src), attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        src2 = src + self.dropout1(src2)
        src3 = self.feedforward(src2)
        src3 = src2 + self.dropout2(src3)

        return src3


class TransformerBlock(nn.Module):
    """Transformer block"""

    def __init__(self,
                 hidden_size,
                 nlayers,
                 ntokens,
                 nhead=8,
                 dropout=0.1,
                 dropatt=0.1,
                 relative_bias=True,
                 pos_emb=False,
                 pad=0):
        """Initialization.

        Args:
          hidden_size: dimension of inputs and hidden states
          nlayers: number of layers
          ntokens: number of output categories
          nhead: number of self-attention heads
          dropout: dropout rate
          dropatt: drop attention rate
          relative_bias: bool, indicate whether use a relative position based
            attention bias
          pos_emb: bool, indicate whether use a learnable positional embedding
          pad: pad token index
        """

        super(TransformerBlock, self).__init__()

        self.drop = nn.Dropout(dropout)

        self.emb = nn.Embedding(ntokens, hidden_size)
        if pos_emb:
            self.pos_emb = nn.Embedding(500, hidden_size)

        self.layers = nn.ModuleList([
            TransformerEncoder(hidden_size, nhead, hidden_size * 4, dropout,
                               dropatt=dropatt, relative_bias=relative_bias)
            for _ in range(nlayers)])

        self.norm = nn.LayerNorm(hidden_size)

        self.output_layer = nn.Linear(hidden_size, ntokens)
        self.output_layer.weight = self.emb.weight

        self.init_weights()

        self.nlayers = nlayers
        self.nhead = nhead
        self.ntokens = ntokens
        self.hidden_size = hidden_size
        self.pad = pad

    def init_weights(self):
        """Initialize token embedding and output bias."""
        initrange = 0.1
        self.emb.weight.data.uniform_(-initrange, initrange)
        if hasattr(self, 'pos_emb'):
            self.pos_emb.weight.data.uniform_(-initrange, initrange)
        self.output_layer.bias.data.fill_(0)

    def visibility(self, x, device):
        """Mask pad tokens."""
        visibility = (x != self.pad).float()
        visibility = visibility[:, None, :].expand(-1, x.size(1), -1)
        visibility = torch.repeat_interleave(visibility, self.nhead, dim=0)
        return visibility.log()

    def encode(self, x, pos):
        """Standard transformer encode process."""
        h = self.emb(x)
        if hasattr(self, 'pos_emb'):
            h = h + self.pos_emb(pos)
        h_list = []
        visibility = self.visibility(x, x.device)

        for i in range(self.nlayers):
            h_list.append(h)
            h = self.layers[i](
                h.transpose(0, 1), key_padding_mask=visibility).transpose(0, 1)

        output = h
        h_array = torch.stack(h_list, dim=2)

        return output, h_array

    def forward(self, x, pos):
        """Pass the input through the encoder layer.

        Args:
          x: input tokens (required).
          pos: position for each token (optional).
        Returns:
          output: probability distributions for missing tokens.
          state_dict: parsing results and raw output
        """

        batch_size, length = x.size()

        raw_output, _ = self.encode(x, pos)
        raw_output = self.norm(raw_output)
        raw_output = self.drop(raw_output)

        output = self.output_layer(raw_output)
        return output.view(batch_size * length, -1), {'raw_output': raw_output, }


class HSIModual(nn.Module):
    def __init__(self,
                 hidden_size,
                 nlayers=2,
                 nhead=8,
                 conv_size=9,
                 weight_act='softmax'):
        """Initialization.
        Args:
          hidden_size: dimension of inputs and hidden states
          nlayers: number of layers
          nhead: number of self-attention heads
          conv_size: convolution kernel size for parser
          weight_act: relations distribution activation function
        """
        super().__init__()

        self.parser_layers = nn.ModuleList([
            nn.Sequential(Conv1d(hidden_size, conv_size),
                          nn.LayerNorm(hidden_size, elementwise_affine=False),
                          nn.Tanh()) for i in range(nlayers)])

        self.distance_ff = nn.Sequential(
            Conv1d(hidden_size, 2),
            nn.LayerNorm(hidden_size, elementwise_affine=False), nn.Tanh(),
            nn.Linear(hidden_size, 1))

        self.height_ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size, elementwise_affine=False), nn.Tanh(),
            nn.Linear(hidden_size, 1))

        self.relations = ['head', 'child']
        n_rel = len(self.relations)
        self._rel_weight = nn.Parameter(torch.zeros((nlayers, nhead, n_rel)))
        self._rel_weight.data.normal_(0, 0.1)

        self._scaler = nn.Parameter(torch.zeros(2))

        self.n_parse_layers = nlayers
        self.weight_act = weight_act
        self.pad = 0

    @property
    def scaler(self):
        return self._scaler.exp()

    @property
    def rel_weight(self):
        if self.weight_act == 'sigmoid':
            return torch.sigmoid(self._rel_weight)
        elif self.weight_act == 'softmax':
            return torch.softmax(self._rel_weight, dim=-1)

    def compute_block(self, distance, height):
        """Compute constituents from distance and height."""
        beta_logits = (distance[:, None, :] - height[:, :, None]) * self.scaler[0]

        gamma = torch.sigmoid(-beta_logits)
        ones = torch.ones_like(gamma)
        indexes = torch.arange(gamma.size()[1])

        block_mask_left = cummin(gamma.tril(-1) + ones.triu(0), reverse=True, max_value=1)
        block_mask_left = block_mask_left - F.pad(block_mask_left[:, :, :-1], (1, 0), value=0)
        block_mask_left.tril_(0)

        block_mask_right = cummin(gamma.triu(0) + ones.tril(-1), exclusive=True, max_value=1)
        block_mask_right = block_mask_right - F.pad(block_mask_right[:, :, 1:], (0, 1), value=0)
        block_mask_right.triu_(0)

        block_p = block_mask_left[:, :, :, None] * block_mask_right[:, :, None, :]
        block = cumsum(block_mask_left).tril(0) + cumsum(block_mask_right, reverse=True).triu(1)

        return block_p, block, indexes

    def compute_head(self, height):
        """Estimate head for each constituent."""
        _, length = height.size()
        head_logits = height * self.scaler[1]
        index = torch.arange(length, device=height.device)

        mask = (index[:, None, None] <= index[None, None, :]) * (
                index[None, None, :] <= index[None, :, None])
        head_logits = head_logits[:, None, None, :].repeat(1, length, length, 1)
        head_logits.masked_fill_(~mask[None, :, :, :], -1e9)

        head_p = torch.softmax(head_logits, dim=-1)

        return head_p

    def parse(self, h, x):
        """Parse input sentence.

        Args:
          h: hidden representations (required).
          x: input tokens (required).
        Returns:
          distance: syntactic distance
          height: syntactic height
        """

        mask = (x != self.pad)
        mask_shifted = F.pad(mask[:, 1:], (0, 1), value=0)

        height = self.height_ff(h).squeeze(-1)
        height.masked_fill_(~mask, -1e9)

        distance = self.distance_ff(h).squeeze(-1)
        distance.masked_fill_(~mask_shifted, 1e9)

        # Calbrating the distance and height to the same level
        length = distance.size(1)
        height_max = height[:, None, :].expand(-1, length, -1)
        height_max = torch.cummax(height_max.triu(0) - torch.ones_like(height_max).tril(-1) * 1e9, dim=-1)[0].triu(0)

        margin_left = torch.relu(F.pad(distance[:, :-1, None], (0, 0, 1, 0), value=1e9) - height_max)
        margin_right = torch.relu(distance[:, None, :] - height_max)
        margin = torch.where(margin_left > margin_right, margin_right, margin_left).triu(0)

        margin_mask = torch.stack([mask_shifted] + [mask] * (length - 1), dim=1)
        margin.masked_fill_(~margin_mask, 0)
        margin = margin.max()

        distance = distance - margin

        return distance, height

    def forward(self, h, input_ids, distance_delta=None, height_delta=None):
        bsz, length, _ = h.size()

        distance, height = self.parse(h, input_ids)
        if distance_delta != None:
            distance += distance_delta
        if height_delta != None:
            height += height_delta

        eye = torch.eye(length, device=h.device, dtype=torch.bool)
        eye = eye[None, :, :].expand((bsz, -1, -1))

        block_p, block, indexes = self.compute_block(distance, height)
        head_p = self.compute_head(height)
        head = torch.einsum('blij,bijh->blh', block_p, head_p)
        head = head.masked_fill(eye, 0)
        child = head.transpose(1, 2)
        cibling = torch.bmm(head, child).masked_fill(eye, 0)

        return distance, height, cibling, head_p, head, block_p, block, indexes


class RepLearnLoss(nn.Module):
    def __init__(self, loss_weight=0.3, pretrain_epochs=0,
                 made_n_samples=1, propagate_other=False):
        super(RepLearnLoss, self).__init__()
        self.pretrain_epochs = pretrain_epochs
        self.epoch = 0
        self.propagate_other = propagate_other

        self.made_n_samples = made_n_samples
        self.BCE = nn.BCEWithLogitsLoss(reduction='sum')
        self.CE = nn.CrossEntropyLoss(weight=loss_weight, reduction='sum')

    def get_log_joint_prob_nlg(self, logits, decisions):
        probs = torch.softmax(logits, dim=-1)
        return (decisions * probs).sum(dim=-1).log().sum(dim=-1)

    def get_log_joint_prob(self, logits, decisions):
        probs = torch.sigmoid(logits)
        decisions = decisions.float()
        probs = probs * decisions + (1 - probs) * (1 - decisions)
        return probs.log().sum(dim=-1)

    def forward(self, logits, targets):
        logits_1d = logits.contiguous().view(-1, logits.size(-1))
        targets_1d = targets.contiguous().view(-1)
        nlg1_sup_loss = self.CE(logits_1d, targets_1d)
        nlg2_sup_loss = self.BCE(logits, targets)

        return nlg1_sup_loss + nlg2_sup_loss


class SimLoss(nn.Module):
    def __init__(self, loss_weight=0.2, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SimLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features1, features2, labels=None, mask=None):

        device = (torch.device('cuda')
                  if features1.is_cuda
                  else torch.device('cpu'))

        if len(features1.shape) < 3 or len(features2.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features1.shape) > 3 or len(features2.shape) > 3:
            features1 = features1.view(features1.shape[0], features1.shape[1], -1)
            features2 = features2.view(features2.shape[0], features2.shape[1], -1)

        batch_size = features1.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features1.shape[1]
        contrast_feature = torch.cat(torch.unbind(features1, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features1[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        contrast_count2 = features2.shape[1]
        contrast_feature2 = torch.cat(torch.unbind(features2, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature2 = features2[:, 0]
            anchor_count2 = 1
        elif self.contrast_mode == 'all':
            anchor_feature2 = contrast_feature2
            anchor_count2 = contrast_count2
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        anchor_dot_contrast2 = torch.div(
            torch.matmul(anchor_feature2, contrast_feature2.T),
            self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast + anchor_dot_contrast2, dim=1, keepdim=True)
        logits = anchor_dot_contrast + anchor_dot_contrast2 - logits_max.detach()

        mask = mask.repeat(anchor_count + anchor_count2, contrast_count + contrast_count2)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class StrcutT5ForConditionalGeneration(T5PreTrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
        r"decoder.embed_tokens.weight",
        r"lm_head.weight",
        r"final_logits_bias",
        'ConGAT.layer1.heads.3.fc.weight', 'HSI.parser_layers.0.0.conv.weight',
        'ConGAT.layer1.heads.3.attn_fc.weight', 'HSI.distance_ff.0.conv.bias',
        'HSI.distance_ff.3.weight',
        'HSI.distance_ff.0.conv.weight', 'ConGAT.layer1.heads.1.attn_fc.weight',
        'ConGAT.layer1.heads.2.attn_fc.weight', 'HSI.height_ff.3.weight',
        'DepGAT.layer1.heads.2.fc.weight',
        'HSI.height_ff.0.bias', 'ConGAT.layer2.heads.0.attn_fc.weight',
        'DepGAT.layer1.heads.1.attn_fc.weight',
        'HSI.parser_layers.1.0.conv.bias', 'DepGAT.layer2.heads.0.attn_fc.weight',
        'DepGAT.layer1.heads.0.fc.weight', 'ConGAT.layer1.heads.2.fc.weight',
        'HSI._scaler',
        'HSI.distance_ff.3.bias', 'HSI.parser_layers.0.0.conv.bias',
        'DepGAT.layer1.heads.3.fc.weight',
        'ConGAT.layer1.heads.0.fc.weight', 'ConGAT.layer1.heads.0.attn_fc.weight',
        'HSI.parser_layers.1.0.conv.weight', 'HSI.height_ff.0.weight',
        'HSI._rel_weight', 'HSI.height_ff.3.bias',
        'DepGAT.layer2.heads.0.fc.weight', 'DepGAT.layer1.heads.3.attn_fc.weight',
        'DepGAT.layer1.heads.2.attn_fc.weight', 'ConGAT.layer2.heads.0.fc.weight',
        'DepGAT.layer1.heads.0.attn_fc.weight', 'ConGAT.layer1.heads.1.fc.weight',
        'DepGAT.layer1.heads.1.fc.weight']

    _keys_to_ignore_on_load_unexpected = [
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.model_dim = config.d_model
        self.model = T5Model(config)

        self.config = config

        self.is_post_training = False

        self.model.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.model.init_weights()

        self.HSI = HSIModual(config.hidden_size)

        self.ConGAT = GAT(in_dim=768, hidden_dim=768, out_dim=768, num_heads=4)
        self.DepGAT = GAT(in_dim=768, hidden_dim=768, out_dim=768, num_heads=4)

    def get_encoder(self):
        return self.model.get_encoder()

    def set_post_training(self):
        self.is_post_training = True

    def get_decoder(self):
        return self.model.get_decoder()

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.BoolTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            decoder_head_mask: Optional[torch.FloatTensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:
        """

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                # warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder = self.get_encoder()
            encoder_outputs = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        # Structural modeling
        if not self.is_post_training:
            past_key_values, _, _, _, _ = self.struct_modeling(input_ids, hidden_states, attention_mask)
        else:
            past_key_values, head_p, block, block_p, indexes = self.struct_modeling(input_ids, hidden_states, attention_mask)
            last_span_rep = block.index_select(indexes, hidden_states)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = self.model._shift_right(labels)

        # Decode
        decoder = self.get_decoder()
        decoder_outputs = decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            if not self.is_post_training:
                loss_fct = CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            else:
                # LM
                def_loss_fct = CrossEntropyLoss(ignore_index=-100)
                loss_lm = def_loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
                # Const
                def_loss_sim = SimLoss(loss_weight=0.2)
                loss_sim = def_loss_sim(last_span_rep, block)
                def_loss_rep = RepLearnLoss(loss_weight=0.3)
                loss_con_rep = def_loss_rep(block_p)
                loss_con = loss_sim + loss_con_rep
                # Dep
                loss_dep = def_loss_rep(head_p)
                # Summary
                loss = loss_lm + loss_con + loss_dep

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def build_con_struct(self, dists, seqs, struct_s, struct_e):
        """building the constituency structures

        Args:
          dists: syntactic distances (required).
          seqs: input tokens (required).
          struct_s: start nodes in the structure (required).
          struct_e: end nodes in the structure (required).
        Returns:
          parse_tree: constituency structures.
          struct_s: start nodes in the structure.
          struct_e: end nodes in the structure.
        """

        assert len(dists) >= 0
        assert len(dists) == len(seqs)

        if len(dists) == 1:
            parse_tree = seqs[0]
        else:
            max_dists = max(dists[:-1])
            parse_tree = []
            sub_seqs = []
            sub_dists = []
            for d, w in zip(dists, seqs):
                sub_seqs.append(w)
                sub_dists.append(d)
                if d >= max_dists:
                    struct_s.append(seqs[0])
                    struct_e.append(w)
                    parse_t, struct_s, struct_e = self.build_con_struct(sub_dists, sub_seqs, struct_s, struct_e)
                    parse_tree.append(parse_t)
                    sub_seqs = []
                    sub_dists = []
        return parse_tree, struct_s, struct_e

    def struct_modeling(self, input_ids, hidden_rep, mask):
        bsz, length, hds = hidden_rep.size()
        distance, height, cibling, head_p, head, block_p, block, indexes = self.HSI(hidden_rep, input_ids)
        dep_temp = torch.ones(bsz, length, device=height.device)
        dep_ = torch.arange(0, length, dtype=torch.int, device=height.device) * dep_temp
        dep_ = dep_.type(torch.int)
        head_ = torch.argmax(head, dim=1).type(torch.int)
        length_valid = torch.sum(mask, dim=1)
        head_ = torch.where(head_ > length_valid.unsqueeze(1).repeat(1, length), head_,
                            torch.zeros_like(head_, device=self.device, dtype=torch.int))  # .type(torch.int))

        past_key_list = []
        for idx in range(bsz):
            _, stru_s, stru_e = self.build_con_struct(distance[idx, :length_valid[idx]], dep_[idx, :length_valid[idx]],
                                                      [], [])
            stru_s = torch.tensor(stru_s).type(torch.int).to(self.device)
            stru_e = torch.tensor(stru_e).type(torch.int).to(self.device)
            con_f = dgl.graph((stru_s, stru_e)).to(self.device)
            con_f.ndata['feature'] = hidden_rep[idx, 0:con_f.num_nodes(), :]
            con_rep = self.ConGAT(con_f, con_f.ndata['feature'])

            dep_f = dgl.graph((dep_[idx, :length_valid[idx]], head_[idx, :length_valid[idx]])).to(self.device)
            dep_f.ndata['feature'] = hidden_rep[idx, 0:dep_f.num_nodes(), :]
            dep_rep = self.DepGAT(dep_f, dep_f.ndata['feature'])

            total_graph_node = torch.add(con_rep, dep_rep)
            past_key_list.append(total_graph_node)

        final_past_key_list = []
        for idx in range(bsz):
            if past_key_list[idx].shape[0] <= length:
                pad = nn.ZeroPad2d(padding=(0, 0, length - past_key_list[idx].shape[0], 0))
                process_past_key = pad(past_key_list[idx])
            else:
                process_past_key = torch.index_select(past_key_list[idx], dim=0,
                                                      index=torch.tensor([i for i in range(length)]).to(self.device))
            final_past_key_list.append(process_past_key)

        struct_rep = torch.stack(final_past_key_list)

        struct_rep = struct_rep.view(bsz, length, 12, int(self.config.d_model / 12))
        struct_rep = struct_rep.repeat(24, 1, 1, 1, 1)
        struct_rep = struct_rep.permute([0, 1, 3, 2, 4]).split(4)

        return struct_rep, head_p, block, block_p, indexes

    def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self.model._shift_right(labels)

    def _reorder_cache(self, past_key_values, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past_key_values is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past_key_values

        reordered_decoder_past = ()
        for layer_past_states in past_key_values:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past


class StrcutFTT5ForConditionalGeneration(T5PreTrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
        r"decoder.embed_tokens.weight",
        r"lm_head.weight",
        r"final_logits_bias",
        'ConGAT.layer1.heads.3.fc.weight', 'HSI.parser_layers.0.0.conv.weight',
        'ConGAT.layer1.heads.3.attn_fc.weight', 'HSI.distance_ff.0.conv.bias',
        'HSI.distance_ff.3.weight',
        'HSI.distance_ff.0.conv.weight', 'ConGAT.layer1.heads.1.attn_fc.weight',
        'ConGAT.layer1.heads.2.attn_fc.weight', 'HSI.height_ff.3.weight',
        'DepGAT.layer1.heads.2.fc.weight',
        'HSI.height_ff.0.bias', 'ConGAT.layer2.heads.0.attn_fc.weight',
        'DepGAT.layer1.heads.1.attn_fc.weight',
        'HSI.parser_layers.1.0.conv.bias', 'DepGAT.layer2.heads.0.attn_fc.weight',
        'DepGAT.layer1.heads.0.fc.weight', 'ConGAT.layer1.heads.2.fc.weight',
        'HSI._scaler',
        'HSI.distance_ff.3.bias', 'HSI.parser_layers.0.0.conv.bias',
        'DepGAT.layer1.heads.3.fc.weight',
        'ConGAT.layer1.heads.0.fc.weight', 'ConGAT.layer1.heads.0.attn_fc.weight',
        'HSI.parser_layers.1.0.conv.weight', 'HSI.height_ff.0.weight',
        'HSI._rel_weight', 'HSI.height_ff.3.bias',
        'DepGAT.layer2.heads.0.fc.weight', 'DepGAT.layer1.heads.3.attn_fc.weight',
        'DepGAT.layer1.heads.2.attn_fc.weight', 'ConGAT.layer2.heads.0.fc.weight',
        'DepGAT.layer1.heads.0.attn_fc.weight', 'ConGAT.layer1.heads.1.fc.weight',
        'DepGAT.layer1.heads.1.fc.weight']

    _keys_to_ignore_on_load_unexpected = [
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.model_dim = config.d_model
        self.model = T5Model(config)

        self.config = config

        self.model.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.model.init_weights()

        self.HSI = HSIModual(config.hidden_size)

        self.ConGAT = GAT(in_dim=768, hidden_dim=768, out_dim=768, num_heads=4)
        self.DepGAT = GAT(in_dim=768, hidden_dim=768, out_dim=768, num_heads=4)

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.BoolTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            decoder_head_mask: Optional[torch.FloatTensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:
        """

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                # warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder = self.get_encoder()
            encoder_outputs = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self.model._shift_right(labels)

        # Decode
        decoder = self.get_decoder()
        decoder_outputs = decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def build_con_struct(self, dists, seqs, struct_s, struct_e):
        """building the constituency structures

        Args:
          dists: syntactic distances (required).
          seqs: input tokens (required).
          struct_s: start nodes in the structure (required).
          struct_e: end nodes in the structure (required).
        Returns:
          parse_tree: constituency structures.
          struct_s: start nodes in the structure.
          struct_e: end nodes in the structure.
        """

        assert len(dists) >= 0
        assert len(dists) == len(seqs)

        if len(dists) == 1:
            parse_tree = seqs[0]
        else:
            max_dists = max(dists[:-1])
            parse_tree = []
            sub_seqs = []
            sub_dists = []
            for d, w in zip(dists, seqs):
                sub_seqs.append(w)
                sub_dists.append(d)
                if d >= max_dists:
                    struct_s.append(seqs[0])
                    struct_e.append(w)
                    parse_t, struct_s, struct_e = self.build_con_struct(sub_dists, sub_seqs, struct_s, struct_e)
                    parse_tree.append(parse_t)
                    sub_seqs = []
                    sub_dists = []
        return parse_tree, struct_s, struct_e

    def struct_modeling(self, input_ids, hidden_rep, mask, distance_delta, height_delta):
        bsz, length, hds = hidden_rep.size()
        distance, height, cibling, head_p, head, block_p, block, indexes = self.HSI(hidden_rep, input_ids)
        dep_temp = torch.ones(bsz, length, device=height.device)
        dep_ = torch.arange(0, length, dtype=torch.int, device=height.device) * dep_temp
        dep_ = dep_.type(torch.int)
        head_ = torch.argmax(head, dim=1).type(torch.int)
        length_valid = torch.sum(mask, dim=1)
        head_ = torch.where(head_ > length_valid.unsqueeze(1).repeat(1, length), head_,
                            torch.zeros_like(head_, device=self.device, dtype=torch.int))  # .type(torch.int))

        past_key_list = []
        for idx in range(bsz):
            _, stru_s, stru_e = self.build_con_struct(distance[idx, :length_valid[idx]], dep_[idx, :length_valid[idx]],
                                                      [], [])
            stru_s = torch.tensor(stru_s).type(torch.int).to(self.device)
            stru_e = torch.tensor(stru_e).type(torch.int).to(self.device)
            con_f = dgl.graph((stru_s, stru_e)).to(self.device)
            con_f.ndata['feature'] = hidden_rep[idx, 0:con_f.num_nodes(), :]
            con_rep = self.ConGAT(con_f, con_f.ndata['feature'])

            dep_f = dgl.graph((dep_[idx, :length_valid[idx]], head_[idx, :length_valid[idx]])).to(self.device)
            dep_f.ndata['feature'] = hidden_rep[idx, 0:dep_f.num_nodes(), :]
            dep_rep = self.DepGAT(dep_f, dep_f.ndata['feature'])

            total_graph_node = torch.add(con_rep, dep_rep)
            past_key_list.append(total_graph_node)

        final_past_key_list = []
        for idx in range(bsz):
            if past_key_list[idx].shape[0] <= length:
                pad = nn.ZeroPad2d(padding=(0, 0, length - past_key_list[idx].shape[0], 0))
                process_past_key = pad(past_key_list[idx])
            else:
                process_past_key = torch.index_select(past_key_list[idx], dim=0,
                                                      index=torch.tensor([i for i in range(length)]).to(self.device))
            final_past_key_list.append(process_past_key)

        struct_rep = torch.stack(final_past_key_list)

        struct_rep = struct_rep.view(bsz, length, 12, int(self.config.d_model / 12))
        struct_rep = struct_rep.repeat(24, 1, 1, 1, 1)
        struct_rep = struct_rep.permute([0, 1, 3, 2, 4]).split(4)

        return struct_rep, head_p, block, block_p, indexes

    def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self.model._shift_right(labels)

    def _reorder_cache(self, past_key_values, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past_key_values is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past_key_values

        reordered_decoder_past = ()
        for layer_past_states in past_key_values:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past


class Agent(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(Agent, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.output_dim = output_dim

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim1)
        self.fc2 = nn.Linear(self.hidden_dim1, self.hidden_dim2)
        self.fc3 = nn.Linear(self.hidden_dim2, self.output_dim)
        self.sigmoid = nn.Sigmoid()

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input):
        x = self.fc1(input)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.fc3(x)
        output = self.sigmoid(x)

        return output


class StructFinetuner(nn.Module):
    def __init__(self, lm_location, seq_length, input_dim, hidden_dim1, hidden_dim2, output_dim, lr, device):
        super(StructFinetuner, self).__init__()
        self.lm_location = lm_location

        self.tokenizer = T5Tokenizer.from_pretrained(self.lm_location)
        self.config = T5Config.from_pretrained(self.lm_location)

        self.seq_length = seq_length
        self.agent_height = Agent(input_dim, hidden_dim1, hidden_dim2, output_dim)
        self.agent_distance = Agent(input_dim, hidden_dim1, hidden_dim2, output_dim)

        self.lr = lr
        self.optimizer_height = torch.optim.Adam(self.agent_height.parameters(), lr=self.lr)
        self.optimizer_distance = torch.optim.Adam(self.agent_distance.parameters(), lr=self.lr)

        self.device = torch.device(device)
        self.to(self.device)

    def choose_action(self, input):
        output_height = self.agent_height(input)
        sample_height = torch.normal(mean=0.5, std=0.25, size=(input.size()[0], self.seq_length)).to(self.device)
        prob_height = torch.maximum(output_height, sample_height)
        prob_height = torch.clamp(prob_height, 0.01, 0.99)
        prob_height_lb = torch.ones(output_height.size()).to(self.device) - prob_height
        param_height = torch.cat((prob_height_lb, prob_height), 1)
        action_height = torch.multinomial(param_height, 1).squeeze(1)

        output_distance = self.agent_distance(input)
        sample_distance = torch.normal(mean=0.5, std=0.25, size=(input.size()[0], self.seq_length)).to(self.device)
        prob_distance = torch.maximum(output_distance, sample_distance)
        prob_distance = torch.clamp(prob_distance, 0.01, 0.99)
        prob_distance_lb = torch.ones(output_distance.size()).to(self.device) - prob_distance
        param_distance = torch.cat((prob_distance_lb, prob_distance), 1)
        action_distance = torch.multinomial(param_distance, 1).squeeze(1)

        return prob_height, action_height, prob_distance, action_distance

    def learn(self, prob_height, prob_distance, action_height, action_distance):
        loss_height = F.cross_entropy(input=prob_height.squeeze(1), target=action_height)
        self.optimizer_height.zero_grad()
        loss_height.backward(retain_graph=True)
        self.optimizer_height.step()

        loss_distance = F.cross_entropy(input=prob_distance.squeeze(1), target=action_distance)
        self.optimizer_distance.zero_grad()
        loss_distance.backward(retain_graph=True)
        self.optimizer_distance.step()
