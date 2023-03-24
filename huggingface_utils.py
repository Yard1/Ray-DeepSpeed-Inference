import gc
import os
from collections import defaultdict
from unittest.mock import patch

import deepspeed
import torch
from filelock import FileLock
from transformers import AutoModelForCausalLM
from transformers.modeling_utils import dtype_byte_size
from transformers.utils.hub import convert_file_size_to_int


def shard_checkpoint_contiguous(
    state_dict, max_shard_size="10GB", weights_name: str = "pytorch_model.bin"
):
    """
    Same as transformers.modeling_utils.shard_checkpoint, but shards each layer
    into its own file to mitigate https://github.com/microsoft/DeepSpeed/issues/3084.
    """
    max_shard_size = convert_file_size_to_int(max_shard_size)

    sharded_state_dicts = []
    current_block = {}
    current_block_size = 0
    total_size = 0

    layers = defaultdict(list)
    saved_keys = set()
    for key in state_dict:
        if key.startswith("model.decoder.layers."):
            layer_key = ".".join(key.split(".")[:4])
            layers[layer_key].append(key)

    for keys in layers.values():
        for key in keys:
            weight = state_dict[key]
            weight_size = weight.numel() * dtype_byte_size(weight.dtype)

            current_block[key] = weight
            current_block_size += weight_size
            total_size += weight_size
            saved_keys.add(key)
        sharded_state_dicts.append(current_block)
        current_block = {}
        current_block_size = 0

    for key, weight in state_dict.items():
        if key in saved_keys:
            continue
        weight_size = weight.numel() * dtype_byte_size(weight.dtype)

        # If this weight is going to tip up over the maximal size, we split.
        if current_block_size + weight_size > max_shard_size:
            sharded_state_dicts.append(current_block)
            current_block = {}
            current_block_size = 0

        current_block[key] = weight
        current_block_size += weight_size
        total_size += weight_size

    # Add the last block
    sharded_state_dicts.append(current_block)

    # If we only have one shard, we return it
    if len(sharded_state_dicts) == 1:
        return {weights_name: sharded_state_dicts[0]}, None

    # Otherwise, let's build the index
    weight_map = {}
    shards = {}
    for idx, shard in enumerate(sharded_state_dicts):
        shard_file = weights_name.replace(
            ".bin", f"-{idx+1:05d}-of-{len(sharded_state_dicts):05d}.bin"
        )
        shard_file = shard_file.replace(
            ".safetensors",
            f"-{idx + 1:05d}-of-{len(sharded_state_dicts):05d}.safetensors",
        )
        shards[shard_file] = shard
        for key in shard.keys():
            weight_map[key] = shard_file

    # Add the metadata
    metadata = {"total_size": total_size}
    index = {"metadata": metadata, "weight_map": weight_map}
    return shards, index


def reshard_checkpoint(model_name_or_path, dtype, path_to_save_in):
    """
    Loads a transformers model into CPU memory, reshards and saves it to mitigate
    https://github.com/microsoft/DeepSpeed/issues/3084.
    """
    with FileLock(f"{path_to_save_in}.lock"):
        # We use a done marker file so that the other ranks do not
        # go through the process again.
        done_marker = os.path.join(path_to_save_in, ".done")
        if not os.path.exists(done_marker):
            dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
            with deepspeed.OnDevice(dtype=dtype, device="cpu"):
                model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                )
            with patch(
                "transformers.modeling_utils.shard_checkpoint",
                shard_checkpoint_contiguous,
            ):
                model.save_pretrained(path_to_save_in)
            with open(done_marker, "w"):
                pass
            del model
            gc.collect()
    return path_to_save_in
