import logging

from ..arguments import DistillationArgs, InferenceArgs, TrainingArgs, UnshardingArgs
from ..enums import Mode, TuningMethod
from ..utils import log_rank_0, run_rank_n
from .base import ModelWrapper
from .distillation import ModelWrapperForDistillation
from .finetuning import ModelWrapperForFinetuning
from .peft import ModelWrapperForPEFT
from .pretraining import ModelWrapperForPretraining


_MODEL_CLASS_MAPPING = {
    TuningMethod.pretraining: ModelWrapperForPretraining,
    TuningMethod.full_finetuning: ModelWrapperForFinetuning,
    TuningMethod.lora: ModelWrapperForPEFT,
    TuningMethod.prompt_tuning: ModelWrapperForPEFT,
    TuningMethod.distillation: ModelWrapperForDistillation,
}


def get_model(args: TrainingArgs | InferenceArgs | UnshardingArgs | DistillationArgs, mode: Mode) -> ModelWrapper:
    tuning_method = args.tuning_args.tuning_method

    kwargs = {
        "mode": mode,
        "model_name": args.model_args.model_name,
        "pretrained_config": args.model_args.pretrained_config,
        "model_class": args.model_args.model_class,
        "dtype": args.mixed_precision_args.dtype,
        "efficient_initialization": args.model_args.efficient_initialization,
        "attention_implementation": args.model_args.attention_implementation,
        "moe_implementation": args.model_args.moe_implementation,
        "use_padding_free_transformer": args.model_args.use_padding_free_transformer,
        "tensor_parallel_word_embeddings": args.distributed_args.tensor_parallel_word_embeddings,
        "sequence_parallel": args.distributed_args.sequence_parallel,
        "neft_alpha": args.research_args.neft_alpha,
        "trust_remote_code": args.model_args.trust_remote_code,
        "tokenizer_name": args.tokenizer_args.tokenizer_name,
        "additional_special_tokens": args.tokenizer_args.additional_special_tokens,
    }

    # pretraining model wrapper needs some extra arguments for initialization
    if tuning_method in [TuningMethod.pretraining, TuningMethod.distillation]:
        kwargs["micro_batch_size"] = args.training_parameters.micro_batch_size
        kwargs["sequence_length"] = args.datasets[0].class_args.get("sequence_length")
        kwargs["reset_attention_mask"] = args.model_args.reset_attention_mask
        kwargs["reset_position_ids"] = args.model_args.reset_position_ids

    if tuning_method == TuningMethod.distillation:
        kwargs["teacher_model_name"] = args.teacher_args.model_name
        kwargs["teacher_model_class"] = args.teacher_args.model_class
        kwargs["teacher_model_dtype"] = args.teacher_args.dtype
        kwargs["kl_divergence_method"] = args.teacher_args.kl_divergence_method
        kwargs["kl_divergence_weight"] = args.teacher_args.kl_divergence_weight

    if tuning_method in _MODEL_CLASS_MAPPING:
        return _MODEL_CLASS_MAPPING[tuning_method](**kwargs)

    raise ValueError(f"unexpected tuning_method ({tuning_method})")


@run_rank_n
def log_model(model: ModelWrapper) -> None:
    """print model

    Args:
        model (ModelWrapper): model to print
    """

    log_rank_0(logging.INFO, "------------------------ model ------------------------")
    log_rank_0(logging.INFO, model)
    log_rank_0(logging.INFO, "-------------------- end of model ---------------------")
