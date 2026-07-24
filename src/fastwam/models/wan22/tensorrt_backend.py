from __future__ import annotations

import torch


_BF16_TO_COPY_REGISTERED = False


def prepare_tensorrt_backend() -> None:
    """Load Torch-TensorRT and add its missing BF16 cast converter."""
    global _BF16_TO_COPY_REGISTERED

    import torch_tensorrt  # noqa: F401  # registers the torch.compile backend

    if _BF16_TO_COPY_REGISTERED:
        return

    from torch_tensorrt.dynamo._SourceIR import SourceIR
    from torch_tensorrt.dynamo.conversion import impl
    from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
        ConverterPriority,
        dynamo_tensorrt_converter,
    )

    def supports_bfloat16(node, _settings) -> bool:
        return node.kwargs.get("dtype") == torch.bfloat16

    @dynamo_tensorrt_converter(
        torch.ops.aten._to_copy.default,
        capability_validator=supports_bfloat16,
        priority=ConverterPriority.HIGH,
        supports_dynamic_shapes=True,
    )
    def convert_bfloat16_to_copy(ctx, target, args, kwargs, name):
        return impl.cast.to_copy(
            ctx,
            target,
            SourceIR.ATEN,
            name,
            args[0],
            kwargs.get("dtype", args[0].dtype),
            force_layer=True,
        )

    _BF16_TO_COPY_REGISTERED = True


def compile_options(dtype: torch.dtype) -> dict:
    return {
        "enabled_precisions": {dtype},
        "truncate_double": True,
        "workspace_size": 1 << 30,
        "optimization_level": 3,
        "num_avg_timing_iters": 2,
        "min_block_size": 1,
        "pass_through_build_failures": True,
    }
