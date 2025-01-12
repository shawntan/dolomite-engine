from ...config import CommonConfig


class StickBreakingConfig(CommonConfig):
    model_type = "stickbreaking"

    def __init__(
        self,
        add_qkv_bias: bool = False,
        sb_remainder: bool = True,
        forget_gate: bool = False,
        head_norm: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.sb_remainder = sb_remainder
        self.add_qkv_bias = add_qkv_bias
        self.forget_gate = forget_gate
        self.head_norm = head_norm

        if add_qkv_bias:
            assert not self.add_bias
