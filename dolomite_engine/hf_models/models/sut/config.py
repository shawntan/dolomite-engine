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
        num_layers: int = 12,
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
        sequence_mixer_blocks: list[dict] = None,
        mlp_blocks: list[dict] = None,
        router_aux_loss_coef: float = 0.001,
        tie_word_embeddings: bool = True,
        rope_dim: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            vocab_size,
            max_position_embeddings,
            hidden_size,
            1,  # override num_layers
            embedding_dropout,
            normalization_function,
            layer_norm_epsilon,
            initializer_range,
            use_cache,
            bos_token_id,
            eos_token_id,
            pad_token_id,
            position_embedding_type,
            rope_theta,
            rope_scaling,
            m_emb,
            m_width,
            m_residual,
            init_method,
            sequence_mixer_blocks,
            mlp_blocks,
            router_aux_loss_coef,
            tie_word_embeddings,
            rope_dim,
            **kwargs,
        )
        # self.num_layers = 1
        self.num_iters = num_layers
