# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Apply this recipe's verl patch in every interpreter, including Ray workers.

Ray starts workers as fresh processes, so a patch applied on the driver never
reaches the trainer actor. Python imports ``sitecustomize`` automatically at
startup for any interpreter whose ``PYTHONPATH`` contains this directory, which
is the one hook that covers all of them; ``submit_webvoyager.sh`` sets that up.

Deliberately fail-open: this file also runs in interpreters that have nothing to
do with training (``pip``, the ``ray`` CLI, tooling), so a missing verl or a
patch error must never stop the process. Set
``NEMO_GYM_BROWSER_DISABLE_PATCHES=1`` to skip patching, or
``NEMO_GYM_BROWSER_PATCH_DEBUG=1`` to surface failures.
"""

import os
import sys

if os.environ.get("NEMO_GYM_BROWSER_DISABLE_PATCHES") != "1":
    try:
        import patches

        patches.install()
    except ImportError:
        pass  # verl (or this recipe) is not importable here — nothing to patch.
    except Exception:  # pragma: no cover - never break an unrelated interpreter
        if os.environ.get("NEMO_GYM_BROWSER_PATCH_DEBUG") == "1":
            import traceback

            print("nemo_gym/browser: patch install failed", file=sys.stderr)
            traceback.print_exc()
