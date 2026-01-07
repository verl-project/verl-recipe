import datetime
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from collections import defaultdict
from functools import wraps

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from ray.exceptions import RayActorError, RayTaskError
from ray.util.queue import Queue
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from verl.trainer.ppo.utils import Role

QUEUE_NAME = "fault_manager_queue"


@ray.remote
class TokensDict:
    def __init__(self, auto_save_path=None, save_interval=5):
        self.iteration = 0
        self.index_prompt_tokens = {}
        self._lock = threading.Lock()
        self.auto_save_path = auto_save_path
        self.save_interval = save_interval
        self.saving_thread = None
        self.saved_step_ckpt = {}

    def start_save(self):
        if self.auto_save_path:
            if self.saving_thread is None:
                self.saving_thread = threading.Thread(target=self._auto_save, daemon=True)
                self.saving_thread.start()

    def update_data(self, global_id, new_token_info):
        with self._lock:
            if global_id not in self.index_prompt_tokens:
                self.index_prompt_tokens[global_id] = {}
            for k, v in new_token_info.items():
                if k not in self.index_prompt_tokens[global_id]:
                    self.index_prompt_tokens[global_id][k] = type(v)()
            self.index_prompt_tokens[global_id]["finished"] = new_token_info["finished"]
            self.index_prompt_tokens[global_id]["new_token_ids"].extend(new_token_info["new_token_ids"])

    def set(self, global_id, key, value):
        with self._lock:
            if global_id not in self.index_prompt_tokens:
                self.index_prompt_tokens[global_id] = {}
            self.index_prompt_tokens[global_id][key] = value

    def extend(self, global_id, key, value):
        with self._lock:
            if global_id not in self.index_prompt_tokens:
                self.index_prompt_tokens[global_id] = {}
            if key not in self.index_prompt_tokens[global_id]:
                self.index_prompt_tokens[global_id][key] = []
            self.index_prompt_tokens[global_id][key].extend(value)

    def get(self):
        with self._lock:
            return self.index_prompt_tokens

    def clear(self, latest_model_ckpt_step):
        save_dir, _ = os.path.split(self.auto_save_path)
        global_step_path = os.path.join(save_dir, f"global_step_{self.iteration}.pt")
        while True:
            with self._lock:
                finished = [req_info.get("finished", False) for _, req_info in self.index_prompt_tokens.items()]
                if not finished:
                    break
                if all(finished) and os.path.exists(global_step_path):
                    break
            print(f"[fault_manager][{datetime.datetime.now()}] waiting all reqs to be finished and saved")
            time.sleep(1)

        self.index_prompt_tokens.clear()
        # clear expired tokens ckpt
        for iteration in list(self.saved_step_ckpt.keys()):
            if iteration <= latest_model_ckpt_step:
                if os.path.exists(self.saved_step_ckpt[iteration]):
                    os.remove(self.saved_step_ckpt[iteration])
                self.saved_step_ckpt.pop(iteration)

    def try_load(self):
        with self._lock:
            save_dir = os.path.dirname(self.auto_save_path)
            finished_save_path = os.path.join(save_dir, f"global_step_{self.iteration}.pt")
            if os.path.exists(finished_save_path):
                load_data = torch.load(finished_save_path)
                self.index_prompt_tokens = load_data["tokens"]
                return True
            if os.path.exists(self.auto_save_path):
                load_data = torch.load(self.auto_save_path)
                if load_data["iter"] == self.iteration:
                    self.index_prompt_tokens = load_data["tokens"]
                    return True
            return False

    def update_iter(self, iteration):
        self.iteration = iteration

    def _auto_save(self):
        save_dir, save_file = os.path.split(self.auto_save_path)
        save_dir_tmp = os.path.join(os.path.dirname(self.auto_save_path), "tmp")
        tmp_path = os.path.join(save_dir_tmp, save_file)
        os.makedirs(save_dir_tmp, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)
        while True:
            with self._lock:
                torch.save({"iter": self.iteration, "tokens": self.index_prompt_tokens}, tmp_path)
                os.replace(tmp_path, self.auto_save_path)
                finished = sum(
                    [1 for _, req_info in self.index_prompt_tokens.items() if req_info.get("finished", False)]
                )
                print(f"[fault_manager][{datetime.datetime.now()}] finished_num: {finished}")
                if all([req_info.get("finished", False) for _, req_info in self.index_prompt_tokens.items()]):
                    global_step_path = os.path.join(save_dir, f"global_step_{self.iteration}.pt")
                    shutil.copy(self.auto_save_path, global_step_path)
                    self.saved_step_ckpt[self.iteration] = global_step_path
            time.sleep(self.save_interval)


@ray.remote
class NodeWorker:
    def __init__(self, actor_pids, device_name):
        self.actor_pids = actor_pids
        self.get_usage_fn = self._get_npu_usage if device_name == "npu" else self._get_gpu_usage
        self.get_chip_info_cmd = ["npu-smi", "info"] if device_name == "npu" else ["nvidia-smi"]

    def is_chip_free(self):
        devices_info = set()
        chip_info = self._exec_shell(self.get_chip_info_cmd)
        for pid in self.actor_pids:
            device_info = self._get_middle_str("\n", chip_info, str(pid))
            if device_info:
                devices_info.add(tuple(device_info.split()))

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=len(devices_info)) as executor:
            usages = list(executor.map(self.get_usage_fn, devices_info))
        print(f"[fault_manager][{datetime.datetime.now()}] usages: {usages}")
        return all([usage == 0 for usage in usages])

    def _get_npu_usage(self, device_info):
        try:
            _, npu_id, chip_id, _ = device_info
            chip_info = self._exec_shell(["npu-smi", "info", "-i", npu_id, "-c", chip_id, "-t", "usages"])
            if not chip_info:
                return 0
            *_, usage = self._get_middle_str("Aicore", chip_info, "\n").split()
            return int(usage)
        except Exception as e:
            print(f"[fault_manager][{datetime.datetime.now()}] get npu usage error: {str(e)}")
            return 0

    def _get_gpu_usage(self, device_info):
        try:
            gpu_id, _, _ = device_info
            chip_info = self._exec_shell(
                ["nvidia-smi", "dmon", "-c", "1", "-i", gpu_id, "-s", "u", "--format", "noheader,nounit"]
            )
            if not chip_info:
                return 0
            _, usage, *_ = chip_info.split()
            return int(usage)
        except Exception as e:
            print(f"[fault_manager][{datetime.datetime.now()}] get gpu usage error: {str(e)}")
            return 0

    @staticmethod
    def _exec_shell(cmd: list):
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=True)
            return result.stdout
        except subprocess.CalledProcessError:
            return False

    @staticmethod
    def _get_middle_str(left, text, right):
        middle = "(.*?)" if left and right else "(.*)"
        match = re.search(rf"{left}{middle}{right}", text)
        if match:
            return match.group(1)
        return ""


class FaultMgr:
    trainer = None
    tokens_queue = None
    tokens_dict = None
    request_global_id_map = {}
    node_workers = []
    timeout_chip_check = False
    device_type = "cpu"
    node_pids = defaultdict(list)

    @classmethod
    def init_tokens_queue(cls):
        cls.tokens_queue = Queue(
            actor_options={
                "name": QUEUE_NAME,
                "scheduling_strategy": cls._get_head_node_strategy(),
                # "max_concurrency": 4, # better be the num of vllm servers
            }
        )

    @classmethod
    def bind_trainer(cls, trainer):
        from recipe.fault_recover.agent_loop.fault_recover_agent_loop import (
            FaultRecoverAgentLoopManager as AgentLoopManager,
        )

        print(f"[fault_manager][{datetime.datetime.now()}] start bind trainer")
        cls.trainer = trainer
        cls.tokens_dict = TokensDict.options(scheduling_strategy=cls._get_head_node_strategy()).remote(
            auto_save_path=cls.trainer.config.fault_manager.tokens_save_file,
            save_interval=cls.trainer.config.fault_manager.tokens_save_interval,
        )
        cls.catch_rollout_tokens()
        cls.device_type = cls.trainer.actor_rollout_wg.get_device_name()[0]
        cls.timeout_chip_check = (cls.trainer.config.fault_manager.timeout_chip_free > 0) and cls.device_type != "cpu"

        AgentLoopManager.generate_sequences = cls.catch_rollout_fault(
            cls.timeout(AgentLoopManager.generate_sequences), roles=[Role.ActorRollout, Role.RefPolicy]
        )

        if cls.timeout_chip_check:
            cls._init_node_workers()

    @classmethod
    def reschedule(cls, func):
        @wraps(func)
        def wrapper(config, task_runner_class=None):
            try:
                func(config, task_runner_class)
            except Exception as reschedule_error:
                print(f"[fault_manager][{datetime.datetime.now()}] catch reschedule fault: {reschedule_error}")
                if config.fault_manager.enable:
                    max_reschedule_times = config.fault_manager.max_reschedule_times
                    reschedule_times = 0
                    while (max_reschedule_times > 0) and (reschedule_times < max_reschedule_times):
                        try:
                            ray.shutdown()
                            print(f"[fault_manager][{datetime.datetime.now()}] start reschedule")
                            func(config, task_runner_class)
                        except Exception as e:
                            print(
                                f"[fault_manager][{datetime.datetime.now()}] catch reschedule fault: "
                                f"{e} during recover, reschedule_times: {reschedule_times}/{max_reschedule_times}"
                            )
                            reschedule_error = e
                            reschedule_times += 1
                        else:
                            break
                    else:
                        raise reschedule_error
                else:
                    raise reschedule_error

        return wrapper

    @classmethod
    def rebuild_wg(cls, roles: list):
        if not cls.trainer:
            raise ValueError("[fault_manager] Have not bound trainer!")
        print(f"[fault_manager][{datetime.datetime.now()}] start rebuild wg")
        from verl.single_controller.ray import RayClassWithInitArgs
        from verl.single_controller.ray.base import create_colocated_worker_cls

        for role in roles:
            resource_pool = cls.trainer.resource_pool_manager.get_resource_pool(role)
            role_cls = RayClassWithInitArgs(
                cls=cls.trainer.role_worker_mapping[role],
                config=cls._get_role_config(role),
                role=str(role),
            )
            cls.trainer.resource_pool_to_cls[resource_pool][str(role)] = role_cls

        wg_kwargs = cls._get_wg_kwargs()
        for resource_pool, class_dict in cls.trainer.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = cls.trainer.ray_worker_group_cls(
                resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, **wg_kwargs
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())

            for role in class_dict.keys():
                role_wg = spawn_wg[role]
                setattr(cls.trainer, cls._get_wg_name(role), role_wg)
                getattr(cls.trainer, cls._get_wg_name(role)).init_model()
        if cls.timeout_chip_check:
            cls._init_node_workers()

    @classmethod
    def catch_rollout_tokens(cls):
        print(f"[fault_manager][{datetime.datetime.now()}] start catch rollout tokens")

        @ray.remote(num_cpus=1)
        def run(q, td):
            while True:
                req_info = q.get()
                # print(f"[fault manager] catch tokens {req_info}")
                if isinstance(req_info, tuple):
                    request_id, global_id = req_info
                    cls.request_global_id_map[request_id] = global_id
                elif isinstance(req_info, dict):
                    cls._parse_req_tokens(req_info, td)

        run.remote(cls.tokens_queue, cls.tokens_dict)

    @classmethod
    def catch_rollout_fault(cls, func, roles):
        @wraps(func)
        def wrapper(_self, gen_batch_output):
            if gen_batch_output.meta_info.get("validate"):
                gen_batch_output = func(_self, gen_batch_output)
                return gen_batch_output
            try:
                gen_batch_output = cls._update_gen_batch(gen_batch_output, ray.get(cls.tokens_dict.get.remote()))
                gen_batch_output = func(_self, gen_batch_output)
                return gen_batch_output
            except Exception as rebuild_error:
                print(f"[fault_manager][{datetime.datetime.now()}] catch rollout fault: {rebuild_error}")
                max_rebuild_times = cls.trainer.config.fault_manager.max_rebuild_times
                rebuild_times = 0
                while (max_rebuild_times < 0) or (rebuild_times < max_rebuild_times):
                    try:
                        pre_rebuild_result = cls._pre_rebuild()
                        if pre_rebuild_result is not True:
                            rebuild_error = pre_rebuild_result
                            break

                        print(f"[fault_manager][{datetime.datetime.now()}] start rebuild")
                        cls.rebuild_wg(roles=roles)
                        cls.rebuild_rollout_manager()
                        if cls.trainer._load_checkpoint() != 0:
                            cls.trainer.global_steps += 1
                        gen_batch_output = cls._update_gen_batch(
                            gen_batch_output, ray.get(cls.tokens_dict.get.remote())
                        )
                        print(f"[fault_manager][{datetime.datetime.now()}] retry rollout")
                        gen_batch_output = func(cls.trainer.async_rollout_manager, gen_batch_output)
                        return gen_batch_output
                    except Exception as e:
                        print(
                            f"[fault_manager][{datetime.datetime.now()}] catch rebuild fault: "
                            f"{e} during recover retry, rebuild_times: {rebuild_times}/{max_rebuild_times}"
                        )
                        rebuild_error = e
                        rebuild_times += 1
                raise rebuild_error

        return wrapper

    @classmethod
    def timeout(cls, func):
        @wraps(func)
        def wrapper(_self, prompts):
            timeout_task_check_interval = _self.config.fault_manager.timeout_task_check_interval
            timeout_chip_free = _self.config.fault_manager.timeout_chip_free
            if (
                not _self.config.fault_manager.enable
                or timeout_task_check_interval < 0
                or prompts.meta_info.get("validate")
            ):
                return func(_self, prompts)
            if cls.timeout_chip_check:
                free_flag = threading.Event()
                stop_flag = threading.Event()

                def monitor():
                    start_time = time.time()
                    while not stop_flag.is_set():
                        chips_free = all(ray.get([w.is_chip_free.remote() for w in cls.node_workers]))
                        if chips_free:
                            if not start_time:
                                start_time = time.time()
                            elif time.time() - start_time > timeout_chip_free and not free_flag.is_set():
                                free_flag.set()
                        else:
                            start_time = None
                            if free_flag.is_set():
                                free_flag.clear()
                        time.sleep(1)

                t = threading.Thread(target=monitor, daemon=True)
                t.start()

            def _handle_timeout(signum, frame):
                if cls.timeout_chip_check:
                    if free_flag.is_set():
                        [ray.kill(w) for w in cls.trainer.async_rollout_manager.agent_loop_workers]
                        [ray.kill(rr.server_handle) for rr in cls.trainer.async_rollout_manager.rollout_replicas]
                    else:
                        signal.alarm(timeout_task_check_interval)
                else:
                    raise TimeoutError(f"[fault_manager][{datetime.datetime.now()}] {func} timeout")

            signal.signal(signal.SIGALRM, _handle_timeout)
            try:
                signal.alarm(timeout_task_check_interval)
                return func(_self, prompts)
            except (RayTaskError, RayActorError) as e:
                raise TimeoutError(f"[fault_manager][{datetime.datetime.now()}] {func} timeout") from e
            finally:
                if cls.timeout_chip_check:
                    stop_flag.set()
                    signal.alarm(0)
                    t.join(timeout=2)

        return wrapper

    @classmethod
    def init_index_prompt_tokens(cls, gen_batch_output):
        ray.get(cls.tokens_dict.clear.remote(latest_model_ckpt_step=cls._get_latest_global_steps()))
        ray.get(cls.tokens_dict.update_iter.remote(cls.trainer.global_steps))
        loaded = ray.get(cls.tokens_dict.try_load.remote())
        gen_batch_output.non_tensor_batch["global_id"] = np.array(
            [str(i) for i in range(len(gen_batch_output.non_tensor_batch["prompt"]))], dtype=object
        )
        if not loaded:
            prompts = gen_batch_output.non_tensor_batch["raw_prompt"]
            global_ids = gen_batch_output.non_tensor_batch["global_id"]
            for i, global_id in enumerate(global_ids):
                ray.get(cls.tokens_dict.set.remote(global_id, "raw_prompt", prompts[i]))
        ray.get(cls.tokens_dict.start_save.remote())

    @classmethod
    def _get_wg_name(cls, role):
        return {
            str(Role.ActorRollout): "actor_rollout_wg",
            str(Role.RefPolicy): "ref_policy_wg",
        }.get(role)

    @classmethod
    def _get_wg_kwargs(cls):
        wg_kwargs = {}
        if OmegaConf.select(cls.trainer.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = cls.trainer.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(cls.trainer.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(cls.trainer.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(cls.trainer.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(
                        cls.trainer.config.global_profiler.global_tool_config.nsys, "worker_nsight_options"
                    )
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(
                        cls.trainer.config.global_profiler.global_tool_config.nsys, "worker_nsight_options"
                    )
                )
        wg_kwargs["device_name"] = cls.trainer.device_name
        return wg_kwargs

    @classmethod
    def _get_role_config(cls, role):
        return {
            Role.ActorRollout: cls.trainer.config.actor_rollout_ref,
            Role.RefPolicy: cls.trainer.config.actor_rollout_ref,
        }.get(role)

    @classmethod
    def _parse_req_tokens(cls, req_info, td):
        for req_id, new_token_info in req_info.items():
            if req_id in cls.request_global_id_map:
                # fused for better performance
                ray.get(td.update_data.remote(cls.request_global_id_map[req_id], new_token_info))

    @classmethod
    def _update_gen_batch(cls, gen_batch_output, tokens_dict):
        all_tokens = tokens_dict
        global_ids = gen_batch_output.non_tensor_batch["global_id"]
        all_new_token_ids = []
        all_new_token_length = []
        all_token_finished = []

        for global_id in global_ids:
            token_info = all_tokens.get(global_id, {"new_token_ids": [], "finished": False})
            new_token_ids = token_info.get("new_token_ids", [])
            finished = token_info.get("finished", False)
            all_new_token_ids.append(new_token_ids)
            all_new_token_length.append(len(new_token_ids))
            all_token_finished.append(finished)

        if all([length == 0 for length in all_new_token_length]):
            return gen_batch_output

        gen_batch_output.non_tensor_batch["new_token_ids"] = np.array(all_new_token_ids, dtype=object)
        gen_batch_output.non_tensor_batch["finished"] = np.array(all_token_finished, dtype=bool)
        return gen_batch_output

    @classmethod
    def _get_latest_global_steps(cls):
        from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path

        checkpoint_folder = cls.trainer.config.trainer.default_local_dir
        if not os.path.isabs(checkpoint_folder):
            working_dir = os.getcwd()
            checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
        global_step_folder = find_latest_ckpt_path(checkpoint_folder)
        if global_step_folder:
            return int(global_step_folder.split("global_step_")[-1])
        return 0

    @classmethod
    def rebuild_rollout_manager(cls):
        from recipe.fault_recover.agent_loop.fault_recover_agent_loop import (
            FaultRecoverAgentLoopManager as AgentLoopManager,
        )

        if cls.trainer.config.reward_model.enable and cls.trainer.config.reward_model.enable_resource_pool:
            rm_resource_pool = cls.trainer.resource_pool_manager.get_resource_pool(Role.RewardModel)
        else:
            rm_resource_pool = None

        [ray.kill(w) for w in cls.trainer.async_rollout_manager.agent_loop_workers]
        [ray.kill(rr.server_handle) for rr in cls.trainer.async_rollout_manager.rollout_replicas]

        cls.trainer.async_rollout_manager = AgentLoopManager(
            config=cls.trainer.config, worker_group=cls.trainer.actor_rollout_wg, rm_resource_pool=rm_resource_pool
        )

    @classmethod
    def _init_node_workers(cls):
        [ray.kill(w) for w in cls.node_workers]
        cls.node_pids.clear()
        cls.node_workers.clear()
        for node_id, actor_pid in cls.trainer.actor_rollout_wg.get_node_pids():
            cls.node_pids[node_id].append(actor_pid)
        for node_id, actor_pids in cls.node_pids.items():
            node_worker = NodeWorker.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
            ).remote(actor_pids, device_name=cls.device_type)
            cls.node_workers.append(node_worker)

    @classmethod
    def _pre_rebuild(cls):
        if cls.trainer.global_steps != cls._get_latest_global_steps() + 1:
            return Exception(
                f"[fault_manager][{datetime.datetime.now()}] ckpt of fault step {cls.trainer.global_steps - 1} lost"
            )

        for pool in cls.trainer.resource_pool_to_cls.keys():
            for pg in pool.pgs:
                ray.util.remove_placement_group(pg)
            pool.pgs = None

        while not cls.tokens_queue.empty():
            print(f"[fault_manager][{datetime.datetime.now()}] waiting for tokens queue to be empty...")
            time.sleep(1)

        rebuild_time = time.time()
        timeout_rebuild = cls.trainer.config.fault_manager.timeout_rebuild
        while time.time() - rebuild_time < timeout_rebuild:
            try:
                cls.trainer.resource_pool_manager._check_resource_available()
                return True
            except ValueError as e:
                print(f"[fault_manager][{datetime.datetime.now()}] {str(e)}\nwaiting for resource to be ready...")
                time.sleep(5)
        return Exception(
            f"[fault_manager][{datetime.datetime.now()}] "
            f"timeout waiting for resource to be ready for {timeout_rebuild}s"
        )

    @classmethod
    def _get_head_node_strategy(cls):
        return NodeAffinitySchedulingStrategy(
            node_id=ray.get_runtime_context().get_node_id(),
            soft=False,
        )


def get_tokens_queue():
    try:
        tokens_queue = ray.get_actor(QUEUE_NAME)
    except ValueError:
        tokens_queue = None
    return tokens_queue
