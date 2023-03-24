# Based on https://github.com/microsoft/DeepSpeedExamples/tree/master/inference/huggingface/text-generation

import argparse
import gc
import io
import json
import math
import os
from pathlib import Path
from typing import List

import deepspeed
import torch
from deepspeed.runtime.utils import see_memory_usage
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


class DSPipeline:
    """
    Example helper class for comprehending DeepSpeed Meta Tensors, meant to mimic HF pipelines.
    The DSPipeline can run with and without meta tensors.
    """

    def __init__(
        self,
        model_name="bigscience/bloom-3b",
        dtype=torch.float16,
        is_meta=True,
        device=-1,
        checkpoint_path=None,
    ):
        self.model_name = model_name
        self.dtype = dtype

        if isinstance(device, torch.device):
            self.device = device
        elif isinstance(device, str):
            self.device = torch.device(device)
        elif device < 0:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{device}")

        # the Deepspeed team made these so it's super fast to load (~1 minute), rather than wait 10-20min loading time.
        self.tp_presharded_models = [
            "microsoft/bloom-deepspeed-inference-int8",
            "microsoft/bloom-deepspeed-inference-fp16",
        ]

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if is_meta:
            """When meta tensors enabled, use checkpoints"""
            self.config = AutoConfig.from_pretrained(self.model_name)
            self.repo_root, self.checkpoints_json = self._generate_json(checkpoint_path)

            with deepspeed.OnDevice(dtype=torch.float16, device="meta"):
                self.model = AutoModelForCausalLM.from_config(self.config)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

        self.model.eval()

    def __call__(self, inputs=["test"], **kwargs):
        if isinstance(inputs, str):
            input_list = [inputs]
        else:
            input_list = inputs

        outputs = self.generate_outputs(input_list, **kwargs)
        return outputs

    def _generate_json(self, checkpoint_path=None):
        if checkpoint_path is None:
            repo_root = snapshot_download(
                self.model_name,
                allow_patterns=["*"],
                ignore_patterns=["*.safetensors", "*.msgpack", "*.h5"],
                local_files_only=False,
                revision=None,
            )
        else:
            assert os.path.exists(
                checkpoint_path
            ), f"Checkpoint path {checkpoint_path} does not exist"
            repo_root = checkpoint_path

        if os.path.exists(os.path.join(repo_root, "ds_inference_config.json")):
            checkpoints_json = os.path.join(repo_root, "ds_inference_config.json")
        elif self.model_name in self.tp_presharded_models:
            # tp presharded repos come with their own checkpoints config file
            checkpoints_json = os.path.join(repo_root, "ds_inference_config.json")
        else:
            checkpoints_json = "checkpoints.json"

            with io.open(checkpoints_json, "w", encoding="utf-8") as f:
                file_list = [
                    str(entry).split("/")[-1]
                    for entry in Path(repo_root).rglob("*.[bp][it][n]")
                    if entry.is_file()
                ]
                data = {"type": "BLOOM", "checkpoints": file_list, "version": 1.0}
                json.dump(data, f)

        return repo_root, checkpoints_json

    def generate_outputs(self, inputs=["test"], **generate_kwargs):
        input_tokens = self.tokenizer.batch_encode_plus(
            inputs, return_tensors="pt", padding=True
        )
        for t in input_tokens:
            if torch.is_tensor(input_tokens[t]):
                input_tokens[t] = input_tokens[t].to(self.device)

        self.model.cuda().to(self.device)

        outputs = self.model.generate(**input_tokens, **generate_kwargs)
        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return outputs


def init_model(
    args: argparse.Namespace, world_size: int, local_rank: int
) -> DSPipeline:
    """Initialize the deepspeed model"""
    data_type = getattr(torch, args.dtype)

    if local_rank == 0:
        see_memory_usage("before init", True)

    pipe = DSPipeline(
        model_name=args.name,
        dtype=data_type,
        is_meta=args.use_meta_tensor,
        device=local_rank,
        checkpoint_path=args.checkpoint_path,
    )
    if local_rank == 0:
        see_memory_usage("after init", True)
    if args.use_meta_tensor:
        ds_kwargs = dict(base_dir=pipe.repo_root, checkpoint=pipe.checkpoints_json)
    else:
        ds_kwargs = dict()

    gc.collect()
    if args.ds_inference:
        pipe.model = deepspeed.init_inference(
            pipe.model,
            dtype=data_type,
            mp_size=world_size,
            replace_with_kernel_inject=args.use_kernel,
            replace_method=args.replace_method,
            max_tokens=args.max_tokens,
            save_mp_checkpoint_path=args.save_mp_checkpoint_path,
            **ds_kwargs,
        )
    if local_rank == 0:
        see_memory_usage("after init_inference", True)
    return pipe


def generate(
    input_sentences: List[str], pipe: DSPipeline, batch_size: int, **generate_kwargs
) -> List[str]:
    """Generate predictions using a DSPipeline"""
    if batch_size > len(input_sentences):
        # dynamically extend to support larger bs by repetition
        input_sentences *= math.ceil(batch_size / len(input_sentences))

    inputs = input_sentences[:batch_size]
    outputs = pipe(inputs, **generate_kwargs)
    return outputs
