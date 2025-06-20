# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import torch
from dataclasses import dataclass
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss, L1Loss
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.utils import ModelOutput
from transformers import LlamaPreTrainedModel, LlamaForCausalLM
from transformers.models.llama import modeling_llama
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask

from . import utils_graphgpt
from src.utils.modules_utils import MLP
from src.utils.loss_utils import auc_loss


@dataclass
class DoubleHeadsModelOutput(ModelOutput):
    """
    Base class for outputs of models with pretrain task and Sequence Classification task.

    Args:
        pretrain_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `pretrain_labels` is provided):
            Language modeling loss.
        task_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `task_labels` is provided):
            E.g, graph-level classification loss.
        pretrain_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        task_logits (`torch.FloatTensor` of shape `(batch_size, num_choices)`):
            Prediction scores of the multiple choice classification head (scores for each choice before SoftMax).
        past_key_values (`Tuple[Tuple[torch.Tensor]]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of length `config.n_layers`, containing tuples of tensors of shape `(batch_size, num_heads,
            sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            GPT2Attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    pretrain_loss: Optional[torch.FloatTensor] = None
    task_loss: Optional[torch.FloatTensor] = None
    pretrain_logits: torch.FloatTensor = None
    task_logits: torch.FloatTensor = None
    head1_loss: Optional[torch.FloatTensor] = None
    head2_loss: Optional[torch.FloatTensor] = None
    head1_logits: torch.FloatTensor = None
    head2_logits: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class StackedFeatAggregation(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.stacked_feat = config.stacked_feat
        if config.stacked_feat_agg_method == "gated":
            self.weight = nn.Parameter(
                torch.empty((config.stacked_feat, config.hidden_size))
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # WHEN loading from pre-trained with HF's `model.from_pretrained`,
        # IF fails due to name mismatch and etc, the params won't be init properly
        # i.e., contains NAN, 0 and exteremely small vals, e.g., 1.4013e-45
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        if self.config.stacked_feat_agg_method == "gated":
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        if self.config.stacked_feat_agg_method == "gated":
            x = torch.einsum("nsfd,fd->nsd", x, self.weight)
        else:
            x = torch.sum(x, dim=-2)
        return x

    def extra_repr(self) -> str:
        if self.config.stacked_feat_agg_method == "gated":
            repr_ = f"stacked_feat={self.stacked_feat}, hidden_size={self.hidden_size}"
        else:
            repr_ = "sum(x, dim=-2)"
        return repr_


class GraphGPTCausal(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        # 1. Transformer's backbone
        LlamaModel = (
            utils_graphgpt.LlamaModel
            if _use_dropout(self.config) # False
            else modeling_llama.LlamaModel
        )
        if not config.causal_attention:   #     causal_attention = 0 if task_type == "pretrain-mlm" else causal_attention
            print(
                f"\nMonkey Patch {LlamaModel.__name__}'s method `_update_causal_mask`!\n"
            )
            LlamaModel._update_causal_mask = _update_causal_mask # def _update_causal_mask(self, attention_mask, input_tensor, **kwargs): _prepare_4d_attention_mask(attention_mask, input_tensor.dtype)
            # Creates a non-causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
            # `(batch_size, key_value_length)`

        self.model = LlamaModel(config)


        # 1.1 Embedding dropout
        if config.embed_pdrop > 0:  # False
            self.embed_dropout = nn.Dropout(p=config.embed_pdrop)
        else:
            self.embed_dropout = None



        # 1.2 Node/edge attributes stacking
        if config.stack_method in {"short", "long"}:
            self.stacked_feat_agg = StackedFeatAggregation(config) # PlaceHolder


        # 1.3 inputs got raw embed feature
        if config.embed_dim > 0:
            self.raw_embed_dropout = None
            if config.embed_pdrop > 0:
                self.raw_embed_dropout = nn.Dropout(p=config.embed_pdrop)
            self.embed_layernorm = modeling_llama.LlamaRMSNorm(
                config.embed_dim, eps=config.rms_norm_eps
            )
            std = self.config.initializer_range
            self.emb_mask_token = torch.nn.Parameter(
                torch.empty((1, 1, config.embed_dim)).normal_(mean=0.0, std=std),
                requires_grad=True,
            )
            # init_values = 1. / config.embed_dim
            # self.lambda_embed = torch.nn.Parameter(
            #     init_values * torch.ones(config.embed_dim), requires_grad=True
            # )
            # layer-scale with `lambda_embed` produce worse results than LN, and slower training speed
            self.embed_proj = nn.Linear(
                config.embed_dim, config.hidden_size, bias=False
            )
        # 2. Optimization objective
        if config.stack_method in {"short", "long"}:
            print(
                f"Next-token-prediction changed to next/masked-{config.next_n_token}-tokens-prediction!"
            )
            if self.config.next_n_token > 1:
                self.next_n_token_head = nn.Linear(
                    config.hidden_size,
                    config.hidden_size * config.next_n_token,
                    bias=False,
                )
            else:
                self.next_n_token_head = nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        eulerian_position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        inputs_raw_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        label_mask: Optional[torch.Tensor] = None,
        prior: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # 1.1 Converting tokens to look-up embeddings
        # print('begin model ---' * 50)
        # print("input_embeds", inputs_embeds)
        assert inputs_embeds is None
        inputs_embeds = self.model.embed_tokens(input_ids)
        if self.embed_dropout is not None:
            inputs_embeds = self.embed_dropout(inputs_embeds)
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        if len(input_ids.shape) == 3:
            inputs_embeds = self.stacked_feat_agg(inputs_embeds)
            # [bz, seq, feat, dim] ->[bz, seq, dim]
            assert inputs_embeds.shape[:2] == input_ids.shape[:2]
        # print("input_ids at beginning", input_ids.shape)
        input_ids = None

        # 1.2 Deal with input raw embeddings if any
        if self.config.embed_dim > 0:
            inputs_raw_embeds = inputs_raw_embeds.to(inputs_embeds.dtype)
            # For inputs corresponding to -100 label, its embed shall multiply 1, i.e., the embed will be kept
            # [N, seq, next_n] -> [N, seq, 1]
            embed_mask = (labels == -100).sum(dim=-1, keepdim=True).to(bool)
            # [N, seq, 1] * [1, 1, dim] -> [N, seq, dim]
            mask_part = (~embed_mask).to(inputs_embeds.dtype) * self.emb_mask_token
            # [N, seq, 1] * [N, seq, dim] -> [N, seq, dim]
            non_mask_part = embed_mask.to(inputs_embeds.dtype) * inputs_raw_embeds

            inputs_raw_embeds = non_mask_part + mask_part
            inputs_raw_embeds = self.embed_layernorm(inputs_raw_embeds)
            if self.raw_embed_dropout is not None:
                inputs_raw_embeds = self.raw_embed_dropout(inputs_raw_embeds)
            inputs_raw_embeds = self.embed_proj(inputs_raw_embeds)
            inputs_embeds = inputs_embeds + inputs_raw_embeds
        # print("inputs_embeds", inputs_embeds.shape)
        # print("attention_mask", attention_mask.shape)
        # print("position_ids", position_ids.shape)
        # if past_key_values is not None:
        #     print("past_key_values", past_key_values.shape)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # print(outputs.keys())
        # if 'past_key_values' in outputs.keys():
        #     print("past_key_values in outputs dict:", outputs.past_key_values)
        hidden_states = outputs[0]  # [N, seq, dim]
        # print(hidden_states.shape)
        if self.config.stack_method == "long":
            batch_size, seq, _ = hidden_states.shape  # [N, seq, dim]
            # i). obtain mask
            mask_m = labels != -100  # [N, seq, next_n]
            # ii). deal hidden states: mask and reshape
            # [N, seq, dim] -> [N, seq, dim*next_n] -> [N, seq, next_n, dim]
            hidden_states = self.next_n_token_head(hidden_states).reshape(
                (batch_size, seq, self.config.next_n_token, -1)
            )
            # [N, seq, next_n, dim] -> [M, dim]
            hidden_states = hidden_states[mask_m]
            # iii). deal labels: mask
            # [N, seq, next_n] -> [M]
            labels = labels[mask_m]
            logits = self.lm_head(hidden_states)
        # the version below can save lots of GPU memory and boost speed
        if self.config.stack_method == "short" and labels is not None:
            dim = hidden_states.shape[-1]  # [N, seq, dim]
            # i). obtain mask
            labels_m = labels[:, :, 0]  # [N, seq, next_n] -> [N, seq]
            # print("labels", labels.shape)
            # print("labels_m", labels_m.shape)
            mask_m = labels_m != -100  # [N, seq]
            # print("mask_m", mask_m.shape)
            # print(mask_m)
            # ii). deal hidden states: mask and reshape
            # [N, seq, dim] -> [M, dim]
            hidden_states = hidden_states[mask_m]
            # print("hidden_states", hidden_states.shape)
            # [M, dim] -> [M, dim*next_n]
            hidden_states = self.next_n_token_head(hidden_states)
            # print("hidden_states", hidden_states.shape)
            # [M, dim*next_n] -> [M*next_n, dim]
            hidden_states = hidden_states.reshape((-1, dim))
            # print("hidden_states", hidden_states.shape)
            # iii). deal labels: mask and reshape
            # [N, seq, next_n] -> [M, next_n]
            labels = labels[mask_m]
            # [M, next_n] -> [M*next_n]
            labels = labels.reshape(-1)
            logits = self.lm_head(hidden_states)
            # print("logits", logits.shape)
        if self.config.stack_method == "short" and labels is None:
            dim = hidden_states.shape[-1]  # [N, seq, dim]
            # i). obtain mask
            # ii). deal hidden states: mask and reshape
            # [N, seq, dim] -> [M, dim]
            mask_shape = hidden_states.shape[:-1]
            mask_m = torch.ones(mask_shape, dtype=torch.bool)
            # print(mask_m)
            # print("hidden_states", hidden_states.shape)
            hidden_states = hidden_states[mask_m]
            # [M, dim] -> [M, dim*next_n]
            # print("hidden_states", hidden_states.shape)
            hidden_states = self.next_n_token_head(hidden_states)
            # print("hidden_states", hidden_states.shape)
            # [M, dim*next_n] -> [M*next_n, dim]
            hidden_states = hidden_states.reshape((-1, dim))
            # print("hidden_states", hidden_states.shape)
            # iii). deal labels: mask and reshape
            # [N, seq, next_n] -> [M, next_n]
            logits = self.lm_head(hidden_states)
            logits = logits.unsqueeze(0)
        #     print("logits", logits.shape)
        # print('end model ---' * 50)
        loss = None
        if labels is not None:
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            logits = logits.view(-1, self.config.vocab_size)
            labels = labels.view(-1)
            # Enable model parallelism
            labels = labels.to(logits.device)
            loss = loss_fct(logits.float(), labels)
            # convert logits to float before cross-entropy for molecule datasets like PCQM4M-v2, MOLPCBA and etc.
            # because when batch-size too large, ce with fp16 leads to no decrease of loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    def generate_packed(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 1,
        temperature: float = 1.0,
        top_k: int = 50,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        entire_context: bool = False,
    ):
        """
        Custom generate method to produce packed token sequences.

        Args:
            input_ids (torch.Tensor): Input sequence of shape (batch size, sequence length, num_features).
            attention_mask (Optional[torch.Tensor]): Attention mask of shape (batch size, sequence length).
            max_new_tokens (int): Number of new tokens to generate.
            temperature (float): Sampling temperature.
            top_k (int): Number of top logits to consider for sampling.
            eos_token_id (Optional[int]): End-of-sequence token ID.
            pad_token_id (Optional[int]): Padding token ID.

        Returns:
            torch.Tensor: Generated sequences of shape (batch size, new sequence length, num_features).
        """
        batch_size, seq_len, num_features = input_ids.shape
        generated = input_ids

        for _ in range(max_new_tokens):
            # Forward pass to get logits
            outputs = self(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits  # Shape: (batch size, seq_len * num_features, vocab_size)
            logits = logits[:, -num_features:, :]  # Get logits for the last feature tokens

            # Apply temperature scaling
            logits = logits / temperature

            # Top-k sampling
            if top_k > 0:
                top_k_values, _ = torch.topk(logits, top_k, dim=-1)
                logits[logits < top_k_values[..., -1, None]] = -float('inf')

            # Sample next tokens
            probs = F.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(batch_size, num_features)

            # Append to generated sequence
            next_tokens = next_tokens.unsqueeze(1)  # Shape: (batch size, 1, num_features)
            generated = torch.cat((generated, next_tokens), dim=1)

            # Update input_ids for the next iteration
            if entire_context:
                input_ids = torch.cat((input_ids, next_tokens), dim=1)
            else:
                input_ids = next_tokens

            # Check for EOS token
            if eos_token_id is not None and (next_tokens == eos_token_id).all():
                break

        # Pad sequences if necessary
        if pad_token_id is not None and generated.shape[1] < max_new_tokens + seq_len:
            padding = pad_token_id * torch.ones(
                (batch_size, max_new_tokens + seq_len - generated.shape[1], num_features),
                dtype=torch.long,
                device=generated.device,
            )
            generated = torch.cat((generated, padding), dim=1)

        return generated


class GraphGPTDoubleHeadsModel(LlamaPreTrainedModel):
    """
    Refer to `GPT2DoubleHeadsModel` in transformers/models/gpt2/modeling_gpt2.py
    Merge two models `LlamaForCausalLM` & `LlamaForSequenceClassification` in transformers/models/llama/modeling_llama.py
    """

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        # 1. Transformer's backbone
        LlamaModel = (
            utils_graphgpt.LlamaModel
            if _use_dropout(self.config)
            else modeling_llama.LlamaModel
        )
        if not config.causal_attention:
            print(
                f"\nMonkey Patch {LlamaModel.__name__}'s method `_update_causal_mask`!\n"
            )
            LlamaModel._update_causal_mask = _update_causal_mask
        self.model = LlamaModel(config)
        # 1.1 Embedding dropout
        if config.embed_pdrop > 0:
            self.embed_dropout = nn.Dropout(p=config.embed_pdrop)
        else:
            self.embed_dropout = None
        # 1.2 Node/edge attributes stacking
        if config.stack_method in {"short", "long"}:
            self.stacked_feat_agg = StackedFeatAggregation(config)
        # 1.3 inputs got raw embed feature
        if config.embed_dim > 0:
            self.raw_embed_dropout = None
            if config.embed_pdrop > 0:
                self.raw_embed_dropout = nn.Dropout(p=config.embed_pdrop)
            self.embed_layernorm = modeling_llama.LlamaRMSNorm(
                config.embed_dim, eps=config.rms_norm_eps
            )
            self.embed_proj = nn.Linear(
                config.embed_dim, config.hidden_size, bias=False
            )
        # 2. Init for CausalLM, refer to `LlamaForCausalLM`
        self.vocab_size = config.vocab_size
        if self.config.use_ntp:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # 3. Init for SequenceClassification, refer to `LlamaForSequenceClassification`
        bias = self.config.problem_type == "regression"
        self.num_labels = config.num_labels
        if len(self.config.mlp) > 0:
            self.score = MLP(
                config.hidden_size,
                self.num_labels,
                mlp=self.config.mlp,
                hidden_act=self.config.hidden_act,
                dropout=self.config.dropout,
                bias=bias,
            )
        else:
            self.score = nn.Linear(config.hidden_size, self.num_labels, bias=bias)
        self.pos_weight = None

        self.pooling_method = config.pooling_method  # "last|sum|mean"
        print(f"Pooling in last layer is {self.pooling_method}!")
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def reset_rope_scaling_factor(self, scaling_factor):
        if scaling_factor > 0:
            max_seq_len_cached = 0
            for layer in self.model.layers:
                layer.self_attn.rotary_emb.scaling_factor = scaling_factor
                layer.self_attn.rotary_emb.max_seq_len_cached = max_seq_len_cached
            print(
                f"reset {len(self.model.layers)} layers' rotary_emb scaling_factor: {scaling_factor}, max_seq_len_cached: {max_seq_len_cached}"
            )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        inputs_raw_embeds: Optional[torch.FloatTensor] = None,
        pretrain_labels: Optional[torch.LongTensor] = None,
        task_labels: Optional[torch.LongTensor] = None,
        cls_idx: Optional[torch.LongTensor] = None,
        sample_wgt: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, DoubleHeadsModelOutput]:
        r"""
        Args:
            pretrain_labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            task_labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
            cls_idx (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            sample_wgt (`torch.FloatTensor` of shape `(batch_size,)`, *optional*): weight of sample
        Returns:

        Example:

        ```python
        test
        ```"""

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # 1.1 Converting tokens to look-up embeddings
        assert inputs_embeds is None
        inputs_embeds = self.model.embed_tokens(input_ids)
        if self.embed_dropout is not None:
            inputs_embeds = self.embed_dropout(inputs_embeds)
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        if len(input_ids.shape) == 3:
            inputs_embeds = self.stacked_feat_agg(inputs_embeds)
            # [bz, seq, feat, dim] -> [bz, seq, dim]
            assert inputs_embeds.shape[:2] == input_ids.shape[:2]
            in_ = input_ids[:, :, 0]  # [bz, seq, num_feat] -> [bz, seq]
        else:
            in_ = input_ids  # [bz, seq]
        input_ids = None

        # 1.2 Deal with input raw embeddings if any
        if self.config.embed_dim > 0:
            inputs_raw_embeds = inputs_raw_embeds.to(inputs_embeds.dtype)
            inputs_raw_embeds = self.embed_layernorm(inputs_raw_embeds)
            if self.raw_embed_dropout is not None:
                inputs_raw_embeds = self.raw_embed_dropout(inputs_raw_embeds)
            inputs_raw_embeds = self.embed_proj(inputs_raw_embeds)
            inputs_embeds = inputs_embeds + inputs_raw_embeds

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]  # [N, seq, dim]

        # 1. Calculate loss for pre-train, refer to `LlamaForCausalLM`
        pretrain_loss = None
        pretrain_logits = None
        if self.config.use_ntp:
            pretrain_logits = self.lm_head(hidden_states)
            logits = pretrain_logits.float()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            logits = logits.view(-1, self.config.vocab_size)
            labels = pretrain_labels.view(-1)
            # Enable model parallelism
            labels = labels.to(logits.device)
            pretrain_loss = loss_fct(logits, labels)

        # 2. Calculate loss for task, refer to `LlamaForSequenceClassification`
        logits = self.score(hidden_states)  # [N, seq, num_labels]
        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError(
                "Cannot handle batch sizes > 1 if no padding token is defined."
            )
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if in_ is not None:
                sequence_lengths = (
                    torch.ne(in_, self.config.pad_token_id).sum(-1) - 1
                ).to(logits.device)
            else:
                sequence_lengths = -1

        if self.pooling_method == "last":
            pooled_logits = logits[
                torch.arange(batch_size, device=logits.device), sequence_lengths
            ]  # [N, seq, num_labels] -> [N, num_labels]
        else:
            # other pooling: [N, seq, num_labels] & [N, seq] -> [N, seq, num_labels]
            masked_logits = logits * attention_mask.to(dtype=logits.dtype).unsqueeze(
                dim=-1
            )
            if self.pooling_method == "sum":
                # sum pooling:
                pooled_logits = torch.sum(
                    masked_logits, dim=1, keepdim=False
                )  # [N, num_labels]
            else:
                # mean pooling:
                pooled_logits = torch.mean(
                    masked_logits, dim=1, keepdim=False
                )  # [N, num_labels]

        if self.config.loss_type == "token_ce_intra":
            inv_temperature = 20
            hidden_states = nn.functional.normalize(hidden_states, dim=-1)
            # get the slice index of intra-instance classes
            # [bz] -> [bz, 1] -> [bz, num_labels]
            idx1 = (
                torch.arange(batch_size, device=logits.device)
                .reshape((-1, 1))
                .expand((-1, self.num_labels))
            )
            # [num_labels] -> [1, num_labels] -> [bz, num_labels] & [bz, 1] -> [bz, num_labels]
            idx2 = torch.arange(self.num_labels, device=logits.device).reshape(
                (1, -1)
            ).expand((batch_size, -1)) + cls_idx.reshape((-1, 1))
            local_label_embeddings = hidden_states[idx1, idx2]  # [bz, num_labels, dim]
            # [bz, seq, dim] & [bz, dim, num_labels] -> [bz, seq, num_labels]
            logits = (
                torch.matmul(hidden_states, local_label_embeddings.transpose(-2, -1))
                * inv_temperature
            )
            # above to mat-multiply local_label_embeddings
        if self.config.loss_type in {"token_ce", "token_ce_intra"}:
            # below `pooled_logits` for output of evaluation only
            pooled_logits = logits
        task_loss = None
        if task_labels is not None:
            labels = task_labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                if self.config.loss_type == "l1":
                    loss_fct = L1Loss()
                    # reduction="sum" -> remove to recover lf's best result
                    # refer to: https://github.com/microsoft/Graphormer/blob/main/graphormer/criterions/l1_loss.py#L35C26-L35C41
                else:
                    loss_fct = MSELoss()
                labels = labels.to(dtype=pooled_logits.dtype)
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                    # remove .float() to recover lf's best result!!!
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                if self.config.loss_type in {"token_ce", "token_ce_intra"}:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(
                        logits.view(-1, self.num_labels).float(), labels.view(-1)
                    )
                elif self.config.loss_type == "auc":
                    # [batch, num_labels]
                    logits = pooled_logits.view(-1, self.num_labels)
                    y_pred = logits[:, 1].float() - logits[:, 0].float()  # [batch]
                    loss = auc_loss(y_pred, labels.view(-1), self.config.num_neg)
                else:
                    if sample_wgt is None:
                        loss_fct = CrossEntropyLoss()
                        loss = loss_fct(
                            pooled_logits.view(-1, self.num_labels).float(),
                            labels.view(-1),
                        )
                    else:
                        loss_fct = CrossEntropyLoss(reduction="none")
                        loss = loss_fct(
                            pooled_logits.view(-1, self.num_labels).float(),
                            labels.view(-1),
                        )
                        assert (
                            loss.shape[0] == sample_wgt.shape[0]
                        ), f"{loss.shape[0]} != {sample_wgt.shape[0]}"
                        loss = (
                            loss.float().view(-1) * sample_wgt.float().view(-1)
                        ).sum() / sample_wgt.float().sum()
            elif self.config.problem_type == "multi_label_classification":
                is_labeled = labels == labels
                loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
                loss = loss_fct(pooled_logits[is_labeled], labels[is_labeled])
                # remove `.float()` to avoid force-converting fp16 to fp32
                # labels[is_labeled] will convert tensor `labels` from 2D to 1D
            task_loss = loss
        if not return_dict:
            output = (
                pretrain_loss,
                task_loss,
                pretrain_logits,
                pooled_logits.float(),
            ) + outputs[1:]
            return tuple(ele for ele in output if ele is not None)

        return DoubleHeadsModelOutput(
            pretrain_loss=pretrain_loss,
            task_loss=task_loss,
            pretrain_logits=pretrain_logits,
            task_logits=pooled_logits.float(),
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class GraphGPTUniBiDoubleHeadsModel(LlamaPreTrainedModel):
    """
    Refer to `GPT2DoubleHeadsModel` in transformers/models/gpt2/modeling_gpt2.py
    One Head for Causal NTP task -> Unidirectional attn
    One Head for Regression -> Bidirectional attn
    refer to https://aliyuque.antfin.com/james.zqf/ssqcu1/dexa1q0g8givelio?singleDoc# for details
    """

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = utils_graphgpt.UniBiLlamaModel(config)
        # 1. Init for CausalLM, refer to `LlamaForCausalLM`
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # 2. Init for SequenceClassification, refer to `LlamaForSequenceClassification`
        self.num_labels = config.num_labels
        if len(self.config.mlp) > 0:
            self.score = MLP(
                config.hidden_size,
                self.num_labels,
                mlp=self.config.mlp,
                hidden_act=self.config.hidden_act,
                dropout=self.config.dropout,
            )
        else:
            self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)
        self.pos_weight = None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_mask_bi: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pretrain_labels: Optional[torch.LongTensor] = None,
        task_labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, DoubleHeadsModelOutput]:
        r"""
        Args:
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                2D mask contains both uni- & bi- attn mask
            attention_mask_bi (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                2D mask contains ONLY bi- attn mask
            pretrain_labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            task_labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        Returns:

        Example:

        ```python
        test
        ```"""

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            attention_mask_bi=attention_mask_bi,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]  # [bsz, seq, dim]

        # 1. Calculate loss for pre-train, refer to `LlamaForCausalLM`
        pretrain_logits = self.lm_head(hidden_states)  # [bsz, seq, vocab]
        logits = pretrain_logits.float()  # [bsz, seq, vocab]

        pretrain_loss = None
        if pretrain_labels is not None:
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            logits = logits.view(-1, self.config.vocab_size)  # [bsz*seq, vocab]
            labels = pretrain_labels.view(-1)  # [bsz*seq]
            # Enable model parallelism
            labels = labels.to(logits.device)
            pretrain_loss = loss_fct(logits, labels)

        # 2. Calculate loss for task, refer to `LlamaForSequenceClassification`
        eps = 1e-7
        logits = self.score(hidden_states)  # [bsz, seq, num_labels]
        assert task_labels is not None
        labels = task_labels.to(dtype=logits.dtype, device=logits.device)
        assert (
            self.config.problem_type == "regression"
        ), f"problem_type {self.config.problem_type} NOT implemented"
        loss_fct = MSELoss(reduction="none")
        task_loss = loss_fct(logits, labels)  # [bsz, seq, num_labels]
        bi_mask = attention_mask_bi[:, :, None].to(task_loss.dtype)  # [bsz, seq, 1]
        task_loss = task_loss * bi_mask  # [bsz, seq, num_labels]
        task_loss = task_loss.sum() / (bi_mask.sum() + eps)

        if not return_dict:
            output = (
                pretrain_loss,
                task_loss,
                pretrain_logits,
                logits,
            ) + outputs[1:]
            return tuple(ele for ele in output if ele is not None)

        return DoubleHeadsModelOutput(
            pretrain_loss=pretrain_loss,
            task_loss=task_loss,
            pretrain_logits=pretrain_logits,
            task_logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class GraphGPTDenoisingNTPDoubleHeadsModel(LlamaPreTrainedModel):
    """
    Refer to `GPT2DoubleHeadsModel` in transformers/models/gpt2/modeling_gpt2.py
    One Head for Causal NTP task -> Unidirectional attn
    One Head for Denoising 3d coordinate Regression -> Bidirectional attn
    refer to https://aliyuque.antfin.com/james.zqf/ssqcu1/dexa1q0g8givelio?singleDoc# for details
    """

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        # 1. Init input 3d noise position projection layer
        self.noise_proj = nn.Linear(3, config.hidden_size, bias=False)
        # 2. Init Transformer backbone
        self.model = utils_graphgpt.UniBiLlamaModel(config)
        # 3. Init prediction heads
        ## 3.1 Init for CausalLM, refer to `LlamaForCausalLM`
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        ## 3.2 Init for SequenceClassification, refer to `LlamaForSequenceClassification`
        self.num_labels = config.num_labels
        if len(self.config.mlp) > 0:
            self.denoise = MLP(
                config.hidden_size,
                self.num_labels,
                mlp=self.config.mlp,
                hidden_act=self.config.hidden_act,
                dropout=self.config.dropout,
            )
        else:
            self.denoise = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        input_pos: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_mask_bi: Optional[torch.Tensor] = None,
        boundary_mask_idx: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pretrain_labels: Optional[torch.LongTensor] = None,
        task_labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, DoubleHeadsModelOutput]:
        r"""
        Args:
            input_pos (`torch.Tensor` of shape `(batch_size, sequence_length, 3)`, *optional*):
                noise-added 3d coordinates of atoms
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                2D mask contains both uni- & bi- attn mask
            attention_mask_bi (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                2D mask contains ONLY bi- attn mask
            boundary_mask_idx (`torch.LongTensor` of shape `(n, 3)`, *optional*):
                indices of 3D mask (i.e., [bsz, seq-len, seq-len]) indicating the exact position
                where bi-attn tokens shall not attend to
            pretrain_labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            task_labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        Returns:

        Example:

        ```python
        test
        ```"""
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        # 0. prepare inputs embeds
        assert input_ids is not None
        assert inputs_embeds is None
        inputs_embeds = self.model.embed_tokens(input_ids)
        # [bz, seq] -> [bz, seq, dim]
        assert input_pos is not None
        noise_fd = self.noise_proj(input_pos.to(inputs_embeds.dtype))
        # [bz, seq, 3] -> [bz, seq, dim]
        inputs_embeds = attention_mask_bi[:, :, None] * noise_fd + inputs_embeds
        # [bz, seq, 1] * [bz, seq, dim] + [bz, seq, dim] -> [bz, seq, dim]

        # 0.1 run backbone transformer
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            attention_mask_bi=attention_mask_bi,
            boundary_mask_idx=boundary_mask_idx,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]  # [bsz, seq, dim]

        # 1. Calculate loss for NTP pre-train, refer to `LlamaForCausalLM`
        pretrain_logits = self.lm_head(hidden_states)  # [bsz, seq, vocab]
        logits = pretrain_logits.float()  # [bsz, seq, vocab]

        assert pretrain_labels is not None
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        logits = logits.view(-1, self.config.vocab_size)  # [bsz*seq, vocab]
        labels = pretrain_labels.view(-1)  # [bsz*seq]
        # Enable model parallelism
        labels = labels.to(logits.device)
        pretrain_loss = loss_fct(logits, labels)

        # 2. Calculate loss for denoising task, refer to `LlamaForSequenceClassification`
        eps = 1e-7
        logits = self.denoise(hidden_states)  # [bsz, seq, num_labels]
        assert task_labels is not None
        labels = task_labels.to(dtype=logits.dtype, device=logits.device)
        assert (
            self.config.problem_type == "regression"
        ), f"problem_type {self.config.problem_type} NOT implemented"
        loss_fct = MSELoss(reduction="none")
        denoise_loss = loss_fct(logits, labels)  # [bsz, seq, num_labels]
        bi_mask = attention_mask_bi[:, :, None].to(denoise_loss.dtype)  # [bsz, seq, 1]
        denoise_loss = denoise_loss * bi_mask  # [bsz, seq, num_labels]
        # denoise_loss = denoise_loss.sum() / (bi_mask.sum() + eps)
        cnt_bi = attention_mask_bi.max(dim=-1).values.sum()
        # [bsz, seq] -> [bsz] -> scalar
        denoise_loss = denoise_loss.sum() / (cnt_bi + eps)

        if not return_dict:
            output = (
                pretrain_loss,
                denoise_loss,
                pretrain_logits,
                logits,
            ) + outputs[1:]
            return tuple(ele for ele in output if ele is not None)

        return DoubleHeadsModelOutput(
            pretrain_loss=pretrain_loss,
            task_loss=denoise_loss,
            pretrain_logits=pretrain_logits,
            task_logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class GraphGPTDenoisingRegressionDoubleHeadsModel(LlamaPreTrainedModel):
    """
    Refer to `GPT2DoubleHeadsModel` in transformers/models/gpt2/modeling_gpt2.py
    One Head for Supervised Regression task -> Unidirectional attn
    One Head for Denoising 3d coordinate Regression -> Bidirectional attn
    refer to https://aliyuque.antfin.com/james.zqf/ssqcu1/dexa1q0g8givelio?singleDoc# for details
    """

    def __init__(self, config):
        super().__init__(config)
        # 1. Init input 3d noise position projection layer
        self.noise_proj = nn.Linear(3, config.hidden_size, bias=False)
        # 2. Init Transformer backbone
        self.model = utils_graphgpt.UniBiLlamaModel(config)
        # 3. Init prediction heads
        ## 3.1 Init for Denoising Head
        self.denoise = nn.Linear(config.hidden_size, 3, bias=False)
        ## 3.2 Init for Regression Head, refer to `LlamaForSequenceClassification`
        self.num_labels = config.num_labels
        assert len(self.config.mlp) == 0
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        input_pos: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        attention_mask_bi: Optional[torch.LongTensor] = None,
        boundary_mask_idx: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        denoising_labels: Optional[torch.Tensor] = None,
        regression_labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, DoubleHeadsModelOutput]:
        r"""
        Args:
            input_pos (`torch.Tensor` of shape `(batch_size, sequence_length, 3)`, *optional*):
                noise-added 3d coordinates of atoms
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                2D mask contains both uni- & bi- attn mask
            attention_mask_bi (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                2D mask contains ONLY bi- attn mask
            boundary_mask_idx (`torch.LongTensor` of shape `(n, 3)`, *optional*):
                indices of 3D mask (i.e., [bsz, seq-len, seq-len]) indicating the exact position
                where bi-attn tokens shall not attend to
            denoising_labels (`torch.Tensor` of shape `(batch_size, sequence_length, 3)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            regression_labels (`torch.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        Returns:

        Example:

        ```python
        test
        ```"""
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        # 0. prepare inputs embeds
        assert input_ids is not None
        assert inputs_embeds is None
        inputs_embeds = self.model.embed_tokens(input_ids)
        # [bz, seq] -> [bz, seq, dim]
        assert input_pos is not None
        noise_fd = self.noise_proj(input_pos.to(inputs_embeds.dtype))
        # [bz, seq, 3] -> [bz, seq, dim]
        inputs_embeds = attention_mask_bi[:, :, None] * noise_fd + inputs_embeds
        # [bz, seq, 1] * [bz, seq, dim] + [bz, seq, dim] -> [bz, seq, dim]

        # 0.1 run backbone transformer
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            attention_mask_bi=attention_mask_bi,
            boundary_mask_idx=boundary_mask_idx,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]  # [bsz, seq, dim]

        # 1. Calculate loss for supervised regression task, refer to `LlamaForSequenceClassification`
        logits = self.score(hidden_states)  # [bsz, seq, num_labels]
        assert input_ids is not None
        batch_size = input_ids.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError(
                "Cannot handle batch sizes > 1 if no padding token is defined."
            )
        assert self.config.pad_token_id is not None

        sequence_lengths = (attention_mask.sum(-1) - attention_mask_bi.sum(-1) - 1).to(
            logits.device
        )  # [bsz]

        pooled_logits = logits[
            torch.arange(batch_size, device=logits.device), sequence_lengths
        ]
        # [batch, num_labels]

        assert self.config.problem_type == "regression"
        labels = regression_labels.to(
            dtype=pooled_logits.dtype, device=pooled_logits.device
        )
        if self.config.loss_type == "l1":
            loss_fct = L1Loss()
        else:
            loss_fct = MSELoss()
        if self.num_labels == 1:
            loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
        else:
            loss = loss_fct(pooled_logits, labels)

        # 2. Calculate loss for denoising task
        eps = 1e-7
        logits = self.denoise(hidden_states)  # [bsz, seq, 3]
        assert denoising_labels is not None
        labels = denoising_labels.to(dtype=logits.dtype, device=logits.device)
        # [bsz, seq, 3]
        assert (
            self.config.problem_type == "regression"
        ), f"problem_type {self.config.problem_type} NOT implemented"
        loss_fct = MSELoss(reduction="none")
        denoise_loss = loss_fct(logits, labels)  # [bsz, seq, 3]
        bi_mask = attention_mask_bi[:, :, None].to(denoise_loss.dtype)  # [bsz, seq, 1]
        denoise_loss = denoise_loss * bi_mask  # [bsz, seq, 3]
        # denoise_loss = denoise_loss.sum() / (bi_mask.sum() + eps)
        cnt_bi = attention_mask_bi.max(dim=-1).values.sum()
        # [bsz, seq] -> [bsz] -> scalar
        denoise_loss = denoise_loss.sum() / (cnt_bi + eps)

        if not return_dict:
            output = (
                loss,
                denoise_loss,
                pooled_logits,
                logits,
            ) + outputs[1:]
            return tuple(ele for ele in output if ele is not None)

        return DoubleHeadsModelOutput(
            head1_loss=loss,
            head2_loss=denoise_loss,
            head1_logits=pooled_logits,
            head2_logits=logits,
            logits=pooled_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class GraphGPTForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # 1. transformer backbone
        LlamaModel = (
            utils_graphgpt.LlamaModel
            if _use_dropout(self.config)
            else modeling_llama.LlamaModel
        )
        self.model = LlamaModel(config)
        # 2. mlp for supervised fine-tune tasks
        self.num_labels = config.num_labels
        if len(self.config.mlp) > 0:
            self.score = MLP(
                config.hidden_size,
                self.num_labels,
                mlp=self.config.mlp,
                hidden_act=self.config.hidden_act,
                dropout=self.config.dropout,
            )
        else:
            self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)
        self.pos_weight = None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]  # [batch, seq, dim]
        logits = self.score(hidden_states)  # [batch, seq, num_labels]

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError(
                "Cannot handle batch sizes > 1 if no padding token is defined."
            )
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (
                    torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
                ).to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[
            torch.arange(batch_size, device=logits.device), sequence_lengths
        ]  # [batch, num_labels]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                if self.config.loss_type == "l1":
                    loss_fct = L1Loss()
                    # reduction="sum" -> remove to recover lf's best result
                    # refer to: https://github.com/microsoft/Graphormer/blob/main/graphormer/criterions/l1_loss.py#L35C26-L35C41
                else:
                    loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                    # remove .float() to recover lf's best result!!!
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                if self.config.loss_type == "auc":
                    logits = pooled_logits.view(
                        -1, self.num_labels
                    )  # [batch, num_labels]
                    y_pred = logits[:, 1].float() - logits[:, 0].float()  # [batch]
                    loss = auc_loss(y_pred, labels.view(-1), self.config.num_neg)
                else:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(
                        pooled_logits.view(-1, self.num_labels), labels.view(-1)
                    )
            elif self.config.problem_type == "multi_label_classification":
                is_labeled = labels == labels
                loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
                loss = loss_fct(
                    pooled_logits[is_labeled], labels[is_labeled]
                )  # remove `.float()` to avoid force-converting fp16 to fp32
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


def _use_dropout(config):    # normarly not used
    if (
        sum([config.path_pdrop, config.mlp_pdrop]) > 0
        or config.layer_scale_init_value > 0
    ):
        print("Applying dropout in backbone transformer")
        return True
    else:
        print("NOT Applying dropout in backbone transformer")
        return False


def _update_causal_mask(self, attention_mask, input_tensor, **kwargs):
    return _prepare_4d_attention_mask(attention_mask, input_tensor.dtype)
