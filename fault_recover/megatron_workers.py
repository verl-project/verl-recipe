import os

import ray

from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.device import get_device_name
from verl.workers.megatron_workers import AsyncActorRolloutRefWorker


class AsyncFaultRecoverActorRolloutRefWorker(AsyncActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_device_name(self):
        return get_device_name()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_pid(self):
        return os.getpid()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_node_pids(self):
        return ray.get_runtime_context().get_node_id(), os.getpid()
