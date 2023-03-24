# %%
import os
from argparse import ArgumentParser

import pandas as pd
import ray
import ray.util
from ray.air import Checkpoint, ScalingConfig
from ray.train.batch_predictor import BatchPredictor

from deepspeed_predictor import DeepSpeedPredictor


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument("--name", required=True, type=str, help="model_name")
    parser.add_argument(
        "--num_worker_groups",
        required=True,
        type=int,
        help="Number of prediction worker groups",
    )
    parser.add_argument(
        "--num_gpus_per_worker_group",
        required=True,
        type=int,
        help="Number of GPUs per prediction worker group",
    )
    parser.add_argument(
        "--hf_home",
        required=False,
        default=None,
        type=str,
        help="path to use as Hugging Face cache. If none, will be left as default.",
    )
    parser.add_argument(
        "--checkpoint_path",
        required=False,
        default=None,
        type=str,
        help="model checkpoint path",
    )
    parser.add_argument(
        "--save_mp_checkpoint_path",
        required=False,
        default=None,
        type=str,
        help="save-path to store the new model checkpoint",
    )
    parser.add_argument("--batch_size", default=1, type=int, help="batch size")
    parser.add_argument(
        "--dtype",
        default="float16",
        type=str,
        choices=["float32", "float16", "int8"],
        help="data-type",
    )
    parser.add_argument(
        "--ds_inference", action="store_true", help="enable ds-inference"
    )
    parser.add_argument(
        "--use_kernel", action="store_true", help="enable kernel-injection"
    )
    parser.add_argument(
        "--replace_method",
        required=False,
        default="auto",
        type=str,
        help="replace method['', 'auto']",
    )
    parser.add_argument(
        "--max_tokens",
        default=1024,
        type=int,
        help="maximum tokens used for the text-generation KV-cache",
    )
    parser.add_argument(
        "--max_new_tokens", default=50, type=int, help="maximum new tokens to generate"
    )
    parser.add_argument(
        "--use_meta_tensor",
        action="store_true",
        help="use the meta tensors to initialize model",
    )
    parser.add_argument(
        "--use_cache", default=True, type=bool, help="use cache for generation"
    )
    parser.add_argument(
        "--reshard_checkpoint_path",
        required=False,
        default=None,
        type=str,
        help="Path to store a resharded HF checkpoint to mitigate microsoft/DeepSpeed/issues/3084. If not provided, will not reshard",
    )
    return parser


parser = get_parser()
args = parser.parse_args()

# %%
runtime_env = {"working_dir": os.path.dirname(__file__)}

if args.hf_home:
    os.environ["HF_HOME"] = args.hf_home
    runtime_env["env_vars"] = {"HF_HOME": os.environ["HF_HOME"]}

ray.init(runtime_env=runtime_env)


# %%
import pandas as pd

PREDICT_COLUMN = "predict"

df = pd.DataFrame(
    ["DeepSpeed is", "Test", "Fill me", "How are you"] * 16, columns=[PREDICT_COLUMN]
)
ds = (
    ray.data.from_pandas(df)
    .repartition(args.num_gpus_per_worker_group * 2)
    .random_shuffle()
    .fully_executed()
)

# %%
# This is a scaling config for one worker group.
group_scaling_config = ScalingConfig(
    use_gpu=True,
    num_workers=args.num_gpus_per_worker_group,
    trainer_resources={"CPU": 0},
)
batch_predictor = BatchPredictor.from_checkpoint(
    Checkpoint.from_dict({"args": args}),
    DeepSpeedPredictor,
    scaling_config=group_scaling_config,
)

# %%
pred = batch_predictor.predict(
    ds,
    batch_size=1,
    num_cpus_per_worker=0,
    min_scoring_workers=args.num_worker_groups,
    max_scoring_workers=args.num_worker_groups,
    # Kwargs passed to mode.generate
    do_sample=True,
    temperature=0.9,
    max_length=100,
)

# %%
print(pred.to_pandas())
