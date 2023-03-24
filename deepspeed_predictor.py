import os
import socket
from collections import defaultdict
from contextlib import closing
from datetime import timedelta

import pandas as pd
import ray
import ray.util
import torch.distributed as dist
from ray.air import Checkpoint, ScalingConfig
from ray.train.constants import DEFAULT_NCCL_SOCKET_IFNAME
from ray.train.predictor import Predictor
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from deepspeed_utils import generate, init_model
from huggingface_utils import reshard_checkpoint


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


@ray.remote
class PredictionWorker:
    def __init__(self, args, rank, world_size):
        self.args = args
        self.rank = rank
        self.world_size = world_size

    def get_address_and_port(self):
        """Returns the IP address and a free port on this node."""
        addr = ray.util.get_node_ip_address()
        port = find_free_port()

        return addr, port

    def init_distributed(self, local_rank, local_world_size, master_addr, master_port):
        os.environ["MASTER_ADDR"] = str(master_addr)
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(x) for x in range(local_world_size)]
        )

        if "NCCL_SOCKET_IFNAME" not in os.environ:
            os.environ["NCCL_SOCKET_IFNAME"] = DEFAULT_NCCL_SOCKET_IFNAME

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=self.rank,
            world_size=self.world_size,
            timeout=timedelta(seconds=1800),
        )

        self.local_rank = local_rank

        os.environ["RANK"] = str(self.rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["LOCAL_WORLD_SIZE"] = str(local_world_size)
        os.environ["WORLD_SIZE"] = str(self.world_size)

    def init_model(self):
        if self.args.reshard_checkpoint_path:
            self.args.checkpoint_path = reshard_checkpoint(
                self.args.checkpoint_path or self.args.name,
                self.args.dtype,
                self.args.reshard_checkpoint_path,
            )
        self.generator = init_model(self.args, self.world_size, self.local_rank)

    def generate(self, data, column, **kwargs):
        return generate(
            list(data[column]), self.generator, self.args.batch_size, **kwargs
        )


class DeepSpeedPredictor(Predictor):
    def __init__(self, checkpoint: Checkpoint, scaling_config: ScalingConfig) -> None:
        self.checkpoint = checkpoint
        self.scaling_config = scaling_config
        self.init_worker_group(scaling_config)

    def init_worker_group(self, scaling_config: ScalingConfig):
        self.pg = scaling_config.as_placement_group_factory().to_placement_group()
        prediction_worker_cls = PredictionWorker.options(
            num_cpus=scaling_config.num_cpus_per_worker,
            num_gpus=scaling_config.num_gpus_per_worker,
            resources=scaling_config.additional_resources_per_worker,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=self.pg, placement_group_capture_child_tasks=True
            ),
        )
        args = self.checkpoint.to_dict()["args"]
        self.prediction_workers = [
            prediction_worker_cls.remote(args, i, scaling_config.num_workers)
            for i in range(scaling_config.num_workers)
        ]
        self.prediction_workers_ips_ports = ray.get(
            [
                prediction_worker.get_address_and_port.remote()
                for prediction_worker in self.prediction_workers
            ]
        )
        rank_0_ip, rank_0_port = self.prediction_workers_ips_ports[0]
        ip_dict = defaultdict(list)  # map from node ip to the workers on it
        for i, ip_port in enumerate(self.prediction_workers_ips_ports):
            ip_dict[ip_port[0]].append(i)

        tasks = []
        for rank in range(len(self.prediction_workers)):
            worker = self.prediction_workers[rank]
            local_world_size = len(ip_dict[self.prediction_workers_ips_ports[rank][0]])
            local_rank = ip_dict[self.prediction_workers_ips_ports[rank][0]].index(rank)
            tasks.append(
                worker.init_distributed.remote(
                    local_rank, local_world_size, rank_0_ip, rank_0_port
                )
            )

        ray.get(tasks)
        ray.get([worker.init_model.remote() for worker in self.prediction_workers])

    def _predict_pandas(self, data: "pd.DataFrame", **kwargs) -> "pd.DataFrame":
        data_ref = ray.put(data)
        prediction = ray.get(
            [
                worker.generate.remote(data_ref, column="predict", **kwargs)
                for worker in self.prediction_workers
            ]
        )[0]

        return pd.DataFrame(prediction, columns=["output"])

    @classmethod
    def from_checkpoint(cls, checkpoint: Checkpoint, **kwargs) -> "Predictor":
        return cls(checkpoint=checkpoint, **kwargs)
