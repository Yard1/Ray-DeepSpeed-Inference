# Download the model from our S3 mirror as it's faster

import argparse
import subprocess

import ray
import ray.util.scheduling_strategies


def force_on_node(node_id: str, remote_func_or_actor_class):
    scheduling_strategy = ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
        node_id=node_id, soft=False
    )
    options = {"scheduling_strategy": scheduling_strategy}
    return remote_func_or_actor_class.options(**options)


def run_on_every_node(remote_func_or_actor_class, *remote_args, **remote_kwargs):
    refs = []
    for node in ray.nodes():
        if node["Alive"] and node["Resources"].get("GPU", None):
            refs.append(
                force_on_node(node["NodeID"], remote_func_or_actor_class).remote(
                    *remote_args, **remote_kwargs
                )
            )
    return ray.get(refs)


@ray.remote(num_gpus=1)
def download_model(bucket_uri, path_to_save_in):
    subprocess.run(
        [
            "aws",
            "s3",
            "sync",
            "--quiet",
            bucket_uri,
            path_to_save_in,
        ]
    )


@ray.remote(num_gpus=1)
def mount_nvme():
    subprocess.run(
        'drive_name="${1:-/dev/nvme1n1}"; mount_path="${2:-/nvme}"; set -x; sudo file -s "$drive_name"; sudo apt install xfsprogs -y; sudo mkfs -t xfs "$drive_name"; sudo mkdir "$mount_path" && sudo mount "$drive_name" "$mount_path" && sudo chown -R ray "$mount_path"',
        shell=True,
        check=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("function", type=str, help="function in this file to run")
    parser.add_argument("args", nargs="*", type=str, help="string args to function")
    args = parser.parse_args()

    ray.init()
    if args.function not in globals():
        raise ValueError(f"{args.function} doesn't exist")
    fn = globals()[args.function]
    assert callable(fn) or hasattr(fn, "_function")
    print(f"Running {args.function}({', '.join(args.args)})")
    if hasattr(fn, "_function"):
        run_on_every_node(fn, *args.args)
    else:
        fn(*args.args)
