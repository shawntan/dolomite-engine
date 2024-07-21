import torch
from parameterized import parameterized
from transformers import set_seed

from dolomite_engine.hf_models import AttentionHeadType, PositionEmbeddingType
from dolomite_engine.hf_models.models.moe_dolomite.layer import SparseMoEBlock

from ...test_common import TestCommons


SEED = 1234


class ScatterMoETEst(TestCommons):
    # @parameterized.expand(
    #     TestCommons.make_args_matrix(
    #         TestCommons.get_all_devices(),
    #     )
    # )
    def test_scattermoe(
        self,
    ) -> None:

        set_seed(SEED)
        device = torch.device("cuda")
        attention_head_type: AttentionHeadType = AttentionHeadType.mha
        position_embedding_type: PositionEmbeddingType = PositionEmbeddingType.rope
        input_ids, attention_mask, labels = self.get_dummy_inputs(device)
        config = self.get_moe_test_config(attention_head_type, position_embedding_type)
        config.add_bias = False
        config.n_layer = 1
        naive_model = self.from_config(config).to(device)
        config.use_scattermoe = True
        scatter_model = self.from_config(config).to(device)
        print(naive_model)
        print(scatter_model)

        scatter_model.eval()
        naive_model.eval()
        print(naive_model.load_state_dict(scatter_model.state_dict(), strict=False))
        assert len(naive_model.transformer.h) == len(scatter_model.transformer.h)
        for layer_idx in range(len(naive_model.transformer.h)):
            naive_layer = naive_model.transformer.h[layer_idx]
            scatter_layer = scatter_model.transformer.h[layer_idx]
            c_fc_weights = [e.c_fc.weight.data for e in naive_layer.mlp.experts]
            c_proj_weights = [e.c_proj.weight.data for e in naive_layer.mlp.experts]
            scatter_layer.mlp.experts.c_fc.weight.data[:] = torch.stack(c_fc_weights)
            scatter_layer.mlp.experts.c_proj.weight.data[:] = torch.stack(c_proj_weights)

        naive_output = naive_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        naive_logits = naive_output.logits
        # naive_output = naive_output.loss

        scatter_output = scatter_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        scatter_logits = scatter_output.logits
        # scatter_output = scatter_output.loss

        self.assert_equal_tensors(
            naive_logits,
            scatter_logits,
            False,
            rtol_float32=0,
            atol_float32=3e-7,
            rtol_float16=0,
            atol_float16=3e-7,
            rtol_bfloat16=0,
            atol_bfloat16=1e-4,
        )
