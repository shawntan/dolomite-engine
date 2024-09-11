from ..gpt_dolomite.config import GPTDolomiteConfig


class SBDolomiteConfig(GPTDolomiteConfig):
    model_type = "sb_dolomite"

    def __init__(
        self,
        add_qkv_bias: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.add_qkv_bias = add_qkv_bias
