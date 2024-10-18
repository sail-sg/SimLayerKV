from transformers import AutoModelForCausalLM
import torch
from torch import nn
import inspect
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers import AutoModelForCausalLM, LogitsProcessorList, StoppingCriteriaList
import torch.distributed as dist
import warnings
from transformers.generation import validate_stopping_criteria, GreedySearchEncoderDecoderOutput, GenerateDecoderOnlyOutput, GreedySearchDecoderOnlyOutput, GenerateEncoderDecoderOutput
from transformers.generation.streamers import BaseStreamer
from transformers.generation.beam_search import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
from transformers.generation.stopping_criteria import EosTokenCriteria
from transformers.generation.utils import GenerateNonBeamOutput, GenerateOutput
from transformers.generation.configuration_utils import GenerationConfig, GenerationMode
from transformers.modeling_utils import PreTrainedModel
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.generation.beam_constraints import DisjunctiveConstraint, PhrasalConstraint
import numpy as np
from transformers.generation.utils import logging, NEED_SETUP_CACHE_CLASSES_MAPPING 
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama import modeling_llama
from SimLayerKV_attention_llama import flashforward, ori_forward
from transformers.cache_utils import Cache,DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast,CausalLMOutputWithPast
from torch.nn import CrossEntropyLoss
import torch
logger = logging.get_logger(__name__)

def relative_top_filter(scores: torch.FloatTensor, relative_top: float = 0.1, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1) -> torch.FloatTensor:
    scores_normalized = scores.log_softmax(dim=-1) 
    sorted_logits, sorted_indices = torch.sort(scores_normalized, descending=True)
    min_thresh = sorted_logits[..., min_tokens_to_keep-1] 
    probs_max = torch.max(scores_normalized, dim=-1).values
    probs_thresh = probs_max + np.log(relative_top)
    probs_thresh = torch.min(min_thresh, probs_thresh)
    probs_thresh = probs_thresh.unsqueeze(-1)
    scores_normalized[scores_normalized < probs_thresh] = filter_value
    return scores_normalized

modeling_llama.LlamaSdpaAttention.flashforward = flashforward
modeling_llama.LlamaSdpaAttention.ori_forward = ori_forward

@torch.no_grad()
def _sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool,
    streamer: Optional["BaseStreamer"],
    logits_warper: Optional[LogitsProcessorList],
    relative_top: float = 0.1,
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    r"""
    Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
    can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        logits_processor (`LogitsProcessorList`):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        generation_config ([`~generation.GenerationConfig`]):
            The generation configuration to be used as parametrization of the decoding method.
        synced_gpus (`bool`):
            Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        logits_warper (`LogitsProcessorList`, *optional*):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
            to warp the prediction score distribution of the language modeling head applied before multinomial
            sampling at each generation step. Only required with sampling strategies (i.e. `do_sample` is set in
            `generation_config`)
        model_kwargs:
            Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
            an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
        A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.
    """
    # init values
    pad_token_id = generation_config._pad_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
    do_sample = generation_config.do_sample
    if do_sample is True and not isinstance(logits_warper, LogitsProcessorList):
        raise ValueError(
            "`do_sample` is set to `True`, `logits_warper` must be a `LogitsProcessorList` instance (it is "
            f"{logits_warper})."
        )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    batch_size = input_ids.shape[0]
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

    bos_weights = []
    if 'threshold' in model_kwargs:
        threshold_stream = model_kwargs['threshold']
    while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # prepare variable output controls (note: some models won't accept all output controls)
        model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
        model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

        # forward pass to get next token
        outputs, bos_weights_temp = self(**model_inputs, bos_weights = bos_weights, threshold_stream =threshold_stream, return_dict=True)
        if torch.tensor(bos_weights).sum() == 0:
            bos_weights = bos_weights_temp

        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need

        # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        # (the clone itself is always small)
        # next_token_logits = outputs.logits[:, -1, :].clone()

        # next_token_logits = outputs.logits[:, -1, :]
        _, seq_len, dim = outputs.logits.shape
        next_token_logits = outputs.logits[:, -1, :].clone()


        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)
        if do_sample:
            next_token_scores = logits_warper(input_ids, next_token_scores)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # token selection
        if do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0

        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        del outputs

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return input_ids


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        # self.self_attn = LlamaSdpaAttention(config=config, layer_idx=layer_idx)

        self.self_attn = modeling_llama.LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)
        self.mlp = modeling_llama.LlamaMLP(config)
        self.input_layernorm = modeling_llama.LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = modeling_llama.LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        flash_flag: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        
        if flash_flag == False:
            hidden_states, self_attn_weights, present_key_value, bos_prob = self.self_attn.ori_forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        else:
            hidden_states, self_attn_weights, present_key_value, bos_prob = self.self_attn.flashforward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                # position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs, bos_prob

class LlamaModel(modeling_llama.LlamaPreTrainedModel):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        bos_weights = [],
        threshold_stream = 1,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if (
            use_cache and not isinstance(past_key_values, Cache) and not self.training
        ):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        # position_embeddings = self.rotary_emb(hidden_states, position_ids)
        position_embeddings = None

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        layer_num = -1
        bos_probs = []

        for decoder_layer in self.layers:
            layer_num += 1
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs, bos_prob  = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    threshold_stream = threshold_stream
                )

                try:
                    bos_probs.append(bos_prob.item())
                except:
                    bos_probs.append(bos_prob)

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        hidden_states = self.norm(hidden_states)
        
        # [SimLayerKV]
        if (torch.tensor(bos_probs).sum() != 0) & (torch.tensor(bos_weights).sum()==0):
            # print(hidden_states.shape[1])
            for layer_num in range(len(bos_probs)):
                if threshold_stream< 2:
                    if bos_probs[layer_num] >threshold_stream:
                        # print(layer_num)
                        next_decoder_cache.key_cache[layer_num] = torch.cat([next_decoder_cache.key_cache[layer_num][:, :, 0:4],next_decoder_cache.key_cache[layer_num][:, :, -1024:]], dim=-2)
                        next_decoder_cache.value_cache[layer_num] = torch.cat([next_decoder_cache.value_cache[layer_num][:, :, 0:4],next_decoder_cache.value_cache[layer_num][:, :, -1024:]], dim=-2)
                        past_key_values.key_cache[layer_num] = torch.cat([past_key_values.key_cache[layer_num][:, :, 0:4],past_key_values.key_cache[layer_num][:, :, -1024:]], dim=-2)
                        past_key_values.value_cache[layer_num] = torch.cat([past_key_values.value_cache[layer_num][:, :, 0:4],past_key_values.value_cache[layer_num][:, :, -1024:]], dim=-2)
                        torch.cuda.empty_cache()

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        ), bos_probs

    def minicache_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        bos_weights = [],
        threshold_stream = 1,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if (
            use_cache and not isinstance(past_key_values, Cache) and not self.training
        ):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        # position_embeddings = self.rotary_emb(hidden_states, position_ids)
        position_embeddings = None

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        layer_num = -1
        bos_probs = []

        for decoder_layer in self.layers:
            layer_num += 1
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs, bos_prob  = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

                try:
                    bos_probs.append(bos_prob.item())
                except:
                    bos_probs.append(bos_prob)
                

            hidden_states = layer_outputs[0]
            

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        hidden_states = self.norm(hidden_states)
        # print(hidden_states.shape)
        if hidden_states.shape[1] != 1:
            for layer_num in range(len(next_decoder_cache.key_cache)):
                    if layer_num >= len(next_decoder_cache.key_cache)/2:
                        if layer_num%2==0:
                            # print(layer_num)
                            # print(next_decoder_cache.value_cache[layer_num].shape)
                            _, new_v0, new_v1 = slerp(next_decoder_cache.value_cache[layer_num], next_decoder_cache.value_cache[layer_num+1])
                            _, new_k0, new_k1 = slerp(next_decoder_cache.key_cache[layer_num], next_decoder_cache.key_cache[layer_num+1])
                            next_decoder_cache.key_cache[layer_num] = new_k0
                            next_decoder_cache.value_cache[layer_num] = new_v0
                            next_decoder_cache.key_cache[layer_num+1] = new_k1
                            next_decoder_cache.value_cache[layer_num+1] = new_v1
                            # past_key_values.key_cache[layer_num+1] = new_k1
                            # past_key_values.value_cache[layer_num+1] = new_v1
                        torch.cuda.empty_cache()

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        ), bos_probs


class LlamaForCausalLM(modeling_llama.LlamaPreTrainedModel):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        bos_weights = [],
        threshold_stream = 1,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs, bos_weights = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            bos_weights = bos_weights,
            threshold_stream = threshold_stream,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        ), bos_weights

# [For MiniCache]
def slerp(v0, v1, t=0.6, eps=1e-6, ret_ratio=0.9):
    """
    Perform Spherical Linear Interpolation (SLERP) between two vectors v0 and v1 with interpolation factor t.
    
    Parameters:
    - t: A scalar value in the range [0, 1] that controls the interpolation between v0 and v1.
         When t = 0, the result is v0, and when t = 1, the result is v1.
    - v0: A tensor of shape [bsz, headnum, seqlen, headdim], representing the first layer cache.
    - v1: A tensor of shape [bsz, headnum, seqlen, headdim], representing the second layer cache.
    - eps: A small value to avoid division by zero when dealing with very small angles (default is 1e-6).
    - ret_ratio: Retention ratio based on seqlen.
    
    Returns:
    - result: The interpolated tensor between v0 and v1 using SLERP.
    - new_v0: The SLERP result with certain indices replaced by the original values from v0.
    - new_v1: The SLERP result with certain indices replaced by the original values from v1.
    """
    # Step 1: Normalize the input vectors
    v0_ori = v0
    v1_ori = v1
    v0_norm = v0.norm(dim=-1, keepdim=True)
    v1_norm = v1.norm(dim=-1, keepdim=True)
    v0 = v0 / v0.norm(dim=-1, keepdim=True)
    v1 = v1 / v1.norm(dim=-1, keepdim=True)

    
    # Step 2: Compute the dot product (cosine of the angle between them)

    dot = (v0 * v1).sum(dim=-1, keepdim=True)  # Compute the dot product
    dot = torch.clamp(dot, -1.0, 1.0)  # Clamp the dot product to avoid numerical issues
    
    # bsz x headnum x seqlen x 1
    # print(dot.size(-2))
    # retention = (-dot).topk(k=int(dot.size(-2) * ret_ratio), dim=-2)[1]
    if dot.shape[-2]>1024:
        retention = (-dot.abs()).topk(k=1024, dim=-2)[1]
    else:
        retention = (-dot.abs()).topk(k=int(dot.size(-2) * 0.1), dim=-2)[1]
    # Step 3: Compute the angle between the vectors (theta)
    theta = torch.acos(dot)

    # Step 4: Handle the case where the vectors are almost the same (to avoid division by zero)
    sin_theta = torch.sin(theta)
    mask = sin_theta > eps

    # Step 5: Apply the SLERP formula
    angle_0 = torch.sin((1 - t) * theta) / sin_theta
    angle_1 = torch.sin(t * theta) / sin_theta

    # If sin_theta is very small, fall back to linear interpolation
    angle_0 = torch.where(mask, angle_0, 1.0 - t)
    angle_1 = torch.where(mask, angle_1, t)

    # Compute the final SLERP result
    result = angle_0 * v0 + angle_1 * v1
    new_v0 = result * v0_norm
    new_v1 = result * v1_norm
    del v0
    del v1
    del v0_norm
    del v1_norm
    # del result
    torch.cuda.empty_cache()
    # print(result.norm(dim=-1, keepdim=True))
    # exit()
    retention = retention.expand(-1, -1, -1,v0_ori.size(-1))

    new_v0.scatter_(-2, retention, v0_ori.gather(-2, retention))  # Replace retained indices in new_v0 with original v0
    new_v1.scatter_(-2, retention, v1_ori.gather(-2, retention))  # Replace retained indices in new_v1 with original v1

    # return result, v0_ori, v1_ori

    return result, new_v0, new_v1
