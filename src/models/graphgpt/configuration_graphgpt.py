from typing import Optional, List
from transformers import LlamaConfig


class GraphGPTConfig(LlamaConfig):
    r"""
    This is the configuration class to store the configuration of a [`GraphLlamaModel`]. It is used to instantiate an LLaMA
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the LLaMA-7B.

    Configuration objects inherit from [`LlamaConfig`] and can be used to control the model outputs. Read the
    documentation from [`LlamaConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the LLaMA model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`LlamaModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings(`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        pad_token_id (`int`, *optional*, defaults to 0):
        cls_token_id (`int`, *optional*, defaults to None):
        num_neg (`int`, *optional*, defaults to None): For auc loss ONLY
        Example:
    """

    model_type = "graphgpt"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        tie_word_embeddings=False,
        pooling_method="last",
        causal_attention: bool = True,
        # For transformer backbone's dropout -> embed layer, attn & attn residue, mlp layer
        embd_pdrop: float = 0,
        attn_pdrop: float = 0,
        resid_pdrop: float = 0,
        mlp_pdrop: float = 0,
        # For graphGPT input features stack
        stacked_feat: int = 1,
        stacked_feat_agg_method: str = "sum",
        # For pre-train task
        next_n_token: int = 1,
        # For downstream tasks
        cls_token_id=None,
        mlp: Optional[List[int]] = None,
        dropout: float = 0,
        loss_type: Optional[str] = None,
        num_neg: Optional[int] = None,
        **kwargs,
    ):
        self.next_n_token = next_n_token
        # 1. For dropout in transformer backbone
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.mlp_pdrop = mlp_pdrop
        self.stacked_feat = stacked_feat
        self.stacked_feat_agg_method = stacked_feat_agg_method
        # 2. For downstream tasks
        self.cls_token_id = cls_token_id
        self.mlp = [] if mlp is None else list(mlp)
        assert pooling_method in {"last", "sum", "mean"}
        self.pooling_method = pooling_method
        self.dropout = dropout
        self.loss_type = loss_type
        self.num_neg = num_neg
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
