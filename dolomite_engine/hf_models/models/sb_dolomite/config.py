from ...config import CommonConfig


class GPTDolomiteConfig(CommonConfig):
    model_type = "sb_dolomite"

    def __init__(
        self,
        add_qkv_bias: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.add_qkv_bias = add_qkv_bias
