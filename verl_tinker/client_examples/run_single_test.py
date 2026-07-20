import argparse
import asyncio
import os
import traceback

try:
    from .tasks.math_rl.gsm8k import run_math_rl_gsm8k_test
    from .tasks.math_sft_rl.gsm8k import run_math_sft_rl_gsm8k_test
    from .tasks.opd.deepmath import DEFAULT_TEACHER_MODEL, run_opd_deepmath_test
    from .tasks.sdft.single_task import run_sdft_single_task_test
    from .tasks.sft.no_robots import run_no_robot_direct_sft_test, run_no_robot_test
    from .tasks.sft.tulu3 import run_tulu3_test
    from .tasks.utils import shutdown_server, wait_for_healthz_ready
except ImportError:
    # Direct ``python run_single_test.py`` execution from client_examples.
    from tasks.math_rl.gsm8k import run_math_rl_gsm8k_test
    from tasks.math_sft_rl.gsm8k import run_math_sft_rl_gsm8k_test
    from tasks.opd.deepmath import DEFAULT_TEACHER_MODEL, run_opd_deepmath_test
    from tasks.sdft.single_task import run_sdft_single_task_test
    from tasks.sft.no_robots import run_no_robot_direct_sft_test, run_no_robot_test
    from tasks.sft.tulu3 import run_tulu3_test
    from tasks.utils import shutdown_server, wait_for_healthz_ready

DEFAULT_MODEL_NAME = "Qwen/Qwen3-1.7B"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/"
DEFAULT_API_KEY = "tml-verl-tinker-local"

ALL_TESTS = {
    "sft_tulu3": run_tulu3_test,
    "sft_norobot": run_no_robot_test,
    "sft_norobot_no_rollout": run_no_robot_direct_sft_test,
    "sdft_single_task": run_sdft_single_task_test,
    "rl_gsm8k": run_math_rl_gsm8k_test,
    "sft_rl_gsm8k": run_math_sft_rl_gsm8k_test,
    "opd_deepmath": run_opd_deepmath_test,
}


async def main(
    test_name: str,
    model_name: str,
    tokenizer_name_or_path: str | None,
    base_url: str,
    api_key: str,
    teacher_model_name: str,
) -> int:
    tokenizer_name_or_path = tokenizer_name_or_path or model_name
    if test_name not in ALL_TESTS:
        raise Exception(f"test name: {test_name} is not valid, available tests are: {list(ALL_TESTS.keys())}")

    print(f"Starting to test {test_name}, model_name: {model_name}")
    if tokenizer_name_or_path != model_name:
        print(f"Using tokenizer path: {tokenizer_name_or_path}")

    os.environ["TINKER_BASE_URL"] = base_url
    os.environ["TINKER_API_KEY"] = api_key
    os.environ["TINKER_TEACHER_MODEL"] = teacher_model_name

    print(f"Using Tinker server URL: {base_url}")

    wait_for_healthz_ready(base_url)

    test = ALL_TESTS[test_name]
    success = True
    try:
        await test(base_url, model_name=model_name, tokenizer_name_or_path=tokenizer_name_or_path)
    except Exception as e:
        success = False
        print(f"test failed: {test_name}: {e}")
        traceback.print_exc()
    finally:
        shutdown_server(base_url)

    if success:
        return 0

    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one verl_tinker client example against a Tinker server.")
    parser.add_argument(
        "--test-name",
        choices=sorted(ALL_TESTS),
        default="sft_tulu3",
        help="Client workload to run.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Model name sent to the Tinker Cookbook workload.",
    )
    parser.add_argument(
        "--tokenizer-name-or-path",
        default=None,
        help="Tokenizer path/name override. Defaults to --model-name.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Tinker server base URL.",
    )
    parser.add_argument(
        "--api-key",
        default=DEFAULT_API_KEY,
        help="Tinker API key compatibility value.",
    )
    parser.add_argument(
        "--teacher-model-name",
        default=DEFAULT_TEACHER_MODEL,
        help="Teacher model used by OPD workloads.",
    )
    args = parser.parse_args()

    raise SystemExit(
        asyncio.run(
            main(
                test_name=args.test_name,
                model_name=args.model_name,
                tokenizer_name_or_path=args.tokenizer_name_or_path,
                base_url=args.base_url,
                api_key=args.api_key,
                teacher_model_name=args.teacher_model_name,
            )
        )
    )
