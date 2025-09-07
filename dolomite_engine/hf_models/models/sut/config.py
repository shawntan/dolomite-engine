from transformers import PretrainedConfig

from ....utils import divide_if_divisible
from ...config import CommonConfig


class SUTConfig(CommonConfig):
    model_type = "sut"

    def __init__(
        self,
        vocab_size: int = 50304,
        max_position_embeddings: int = 1024,
        hidden_size: int = 768,
        num_iters: int = 12,
        embedding_dropout: float = 0,
        normalization_function: str = "layernorm",
        layer_norm_epsilon: float = 1e-5,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        bos_token_id: int = 50256,
        eos_token_id: int = 50256,
        pad_token_id: int = 50256,
        position_embedding_type: str = "learned_absolute",
        rope_theta: int = 10000,
        rope_scaling: dict | None = None,
        m_emb: float | None = None,
        m_width: float | None = None,
        m_residual: float | None = None,
        init_method: str = "normal",
        mlp_blocks: list[dict] = None,
        sequence_mixer_blocks: list[dict] = None,
        router_aux_loss_coef: float = 0.001,
        tie_word_embeddings: bool = True,
        rope_dim: int | None = None,
        # SUT specific
        pre_layernorm: bool = True,
        enc_uni_dec_layers: list[int] = [0, 0, 0],
        halting: bool = False,
        shared_kv_cache: bool = False,
        **kwargs,
    ) -> None:
        total_layers = sum(enc_uni_dec_layers)
        if "num_layers" in kwargs:
            del kwargs["num_layers"]
        super().__init__(
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            hidden_size=hidden_size,
            num_layers=total_layers,
            embedding_dropout=embedding_dropout,
            normalization_function=normalization_function,
            layer_norm_epsilon=layer_norm_epsilon,
            initializer_range=initializer_range,
            use_cache=use_cache,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            position_embedding_type=position_embedding_type,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            m_emb=m_emb,
            m_width=m_width,
            m_residual=m_residual,
            init_method=init_method,
            sequence_mixer_blocks=sequence_mixer_blocks,
            mlp_blocks=mlp_blocks,
            router_aux_loss_coef=router_aux_loss_coef,
            tie_word_embeddings=tie_word_embeddings,
            rope_dim=rope_dim,
            **kwargs,
        )

        if mlp_blocks is not None and sequence_mixer_blocks is not None:
            assert len(mlp_blocks) == total_layers, (len(mlp_blocks), enc_uni_dec_layers)
            assert len(sequence_mixer_blocks) == total_layers

        self.num_iters = num_iters
        self.pre_layernorm = pre_layernorm
        self.enc_uni_dec_layers = enc_uni_dec_layers
        self.halting = halting
        self.shared_kv_cache = shared_kv_cache
