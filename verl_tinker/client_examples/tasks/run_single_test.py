import asyncio
import os

from tasks.math_rl.gsm8k import run_math_rl_gsm8k_test
from tasks.math_sft_rl.gsm8k import run_math_sft_rl_gsm8k_test
from tasks.sdft.single_task import run_sdft_single_task_test
from tasks.sft.no_robots import run_no_robot_direct_sft_test, run_no_robot_test
from tasks.sft.tulu3 import run_tulu3_test
from tasks.utils import shutdown_server, wait_for_healthz_ready, wait_for_url


DEFAULT_MODEL_NAME = "Qwen/Qwen3-1.7B"
CI_TOKENIZER_PATH = "/mnt/hdfs/model"

ALL_TESTS = {
    "sft_tulu3": run_tulu3_test,
    "sft_norobot": run_no_robot_test,
    "sft_norobot_no_rollout": run_no_robot_direct_sft_test,
    "sdft_single_task": run_sdft_single_task_test,
    "rl_gsm8k": run_math_rl_gsm8k_test,
    "sft_rl_gsm8k": run_math_sft_rl_gsm8k_test,
}


async def main():
    psm = os.environ.get("SERVER_RAY_SERVE_PROXY_PSM", "")
    test_name = os.environ.get("TEST_NAME", "sft_tulu3")
    model_name = os.environ.get("TINKER_CLIENT_MODEL_NAME", DEFAULT_MODEL_NAME)
    if os.environ.get("TINKER_CI_JOB"):
        tokenizer_name_or_path = CI_TOKENIZER_PATH
    else:
        tokenizer_name_or_path = os.environ.get("TINKER_CLIENT_TOKENIZER_PATH", "") or model_name
    if test_name not in ALL_TESTS:
        raise Exception(f"test name: {test_name} is not valid, available tests are: {list(ALL_TESTS.keys())}")

    print(f"Starting to test {test_name} on psm: {psm}, model_name: {model_name}")
    if tokenizer_name_or_path != model_name:
        print(f"Using tokenizer path: {tokenizer_name_or_path}")

    url = wait_for_url(psm=psm)
    os.environ["TINKER_BASE_URL"] = url
    os.environ["TINKER_API_KEY"] = "tml-verl-tinker-local"

    print(f"got url at: {url}")

    wait_for_healthz_ready(url)

    test = ALL_TESTS[test_name]

    success = True
    try:
        await test(url, model_name=model_name, tokenizer_name_or_path=tokenizer_name_or_path)
    except Exception as e:
        success = False
        print(f"test failed: {test_name}: {e}")
    finally:
        shutdown_server(url)

    if success:
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
