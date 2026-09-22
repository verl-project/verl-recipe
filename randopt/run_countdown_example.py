"""Run a RandOpt Countdown example."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


def _set_default_env() -> None:
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    if Path(cuda_home).is_dir():
        os.environ["CUDA_HOME"] = cuda_home
        os.environ["PATH"] = f"{cuda_home}/bin:{os.environ.get('PATH', '')}"
        _prepend_env_path("CPATH", f"{cuda_home}/include")
        _prepend_env_path("CPLUS_INCLUDE_PATH", f"{cuda_home}/include")
        _prepend_env_path("LIBRARY_PATH", f"{cuda_home}/lib64")
        _prepend_env_path("LD_LIBRARY_PATH", f"{cuda_home}/lib64")

    defaults = {
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "VLLM_DISABLE_FLASHINFER_PREFILL": "1",
        "VLLM_ATTENTION_BACKEND": "TORCH_SDPA",
        "TRANSFORMERS_ATTN_IMPLEMENTATION": "sdpa",
        "VLLM_NO_USAGE_STATS": "1",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)

    cuda_devices = os.environ.get("CUDA_DEVICES")
    if cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices


def _prepend_env_path(name: str, value: str) -> None:
    current = os.environ.get(name)
    os.environ[name] = f"{value}:{current}" if current else value


def _check_dependencies() -> None:
    missing = [name for name in ("verl", "ray", "vllm", "transformers") if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(
            f"Missing Python dependencies: {', '.join(missing)}. "
            "Install them with: pip install verl==0.7.1 vllm ray pandas pyarrow tqdm"
        )


def _repeat_records(records: list[dict], count: int) -> list[dict]:
    if count <= 0:
        return records
    return [dict(records[index % len(records)]) for index in range(count)]


def _write_countdown_data(smoke_dir: Path, train_count: int, val_count: int) -> tuple[Path, Path]:
    records = [
        {"numbers": [1, 2, 3, 4], "target": 10},
        {"numbers": [1, 1, 1, 1], "target": 4},
        {"numbers": [2, 2, 2, 2], "target": 8},
        {"numbers": [3, 3, 3, 3], "target": 12},
        {"numbers": [2, 3, 5, 7], "target": 17},
        {"numbers": [1, 4, 6, 8], "target": 19},
        {"numbers": [2, 4, 6, 8], "target": 20},
        {"numbers": [3, 3, 6, 9], "target": 21},
        {"numbers": [1, 5, 5, 10], "target": 20},
        {"numbers": [2, 2, 7, 9], "target": 20},
        {"numbers": [4, 4, 5, 6], "target": 19},
    ]
    smoke_dir.mkdir(parents=True, exist_ok=True)
    train_file = smoke_dir / "countdown_train.json"
    val_file = smoke_dir / "countdown_val.json"
    train_file.write_text(json.dumps(_repeat_records(records, train_count)))
    val_file.write_text(json.dumps(_repeat_records(records, val_count)))
    return train_file, val_file


def main() -> None:
    _set_default_env()
    _check_dependencies()

    package = __package__ or "randopt"
    main_module = f"{package}.main_randopt"
    worker_extension_cls = f"{package}.worker_extension.WorkerExtension"

    smoke_dir = Path(os.environ.get("SMOKE_DIR", "outputs/randopt_smoke"))
    train_max_samples = os.environ.get("TRAIN_MAX_SAMPLES", "20")
    val_max_samples = os.environ.get("VAL_MAX_SAMPLES", "200")
    train_file = Path(os.environ["TRAIN_FILE"]) if os.environ.get("TRAIN_FILE") else None
    val_file = Path(os.environ["VAL_FILE"]) if os.environ.get("VAL_FILE") else None
    if train_file is None or val_file is None:
        train_file, val_file = _write_countdown_data(smoke_dir, int(train_max_samples), int(val_max_samples))
        print(f"Wrote {train_file} ({train_max_samples} train) and {val_file} ({val_max_samples} val)")

    log_file = Path(os.environ.get("LOG_FILE", smoke_dir / "randopt_smoke.log"))
    log_file.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        main_module,
        f"model.path={os.environ.get('MODEL_PATH', 'Qwen/Qwen2.5-1.5B-Instruct')}",
        "data.task_type=countdown",
        f"data.train_files={train_file}",
        f"data.val_files={val_file}",
        f"data.train_max_samples={train_max_samples}",
        f"data.val_max_samples={val_max_samples}",
        "randopt.sigma=0.001",
        f"randopt.sigma_list={os.environ.get('SIGMA_LIST', 'null')}",
        f"randopt.population_size={os.environ.get('POPULATION_SIZE', '500')}",
        f"randopt.top_k_ratios={os.environ.get('TOP_K_RATIOS', '[0.02,0.1]')}",
        f"randopt.num_engines={os.environ.get('NUM_ENGINES', os.environ.get('N_GPUS', '6'))}",
        f"randopt.tensor_parallel_size={os.environ.get('TP', '1')}",
        "randopt.precision=bfloat16",
        f"randopt.max_tokens={os.environ.get('MAX_TOKENS', '512')}",
        "randopt.temperature=0.0",
        f"randopt.gpu_memory_utilization={os.environ.get('GPU_MEMORY_UTILIZATION', '0.6')}",
        f"randopt.worker_extension_cls={worker_extension_cls}",
        f"randopt.debug_print_samples={os.environ.get('DEBUG_PRINT_SAMPLES', '1')}",
        f"randopt.debug_max_samples={os.environ.get('DEBUG_MAX_SAMPLES', '4')}",
        'trainer.logger=["console"]',
        f"trainer.project_name={os.environ.get('PROJECT_NAME', 'randopt-smoke')}",
        f"trainer.experiment_name={os.environ.get('EXP_NAME', 'countdown-smoke')}",
        f"trainer.n_gpus_per_node={os.environ.get('N_GPUS', '1')}",
        "trainer.nnodes=1",
        f"trainer.default_local_dir={os.environ.get('CKPTS_DIR', str(smoke_dir / 'ckpts'))}",
        f"trainer.total_epochs={os.environ.get('TOTAL_EPOCHS', '1')}",
        "trainer.test_freq=1",
    ]

    with log_file.open("w") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise SystemExit(return_code)
    print(f"Smoke test log: {log_file}")


if __name__ == "__main__":
    main()
