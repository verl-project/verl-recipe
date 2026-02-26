import os
import pickle
import sys

from megatron.training.global_vars import get_args
from megatron.training.initialize import initialize_megatron
from mindspeed.megatron_adaptor import get_mindspeed_args
from mindspeed_mm.arguments import extra_args_provider_decorator
from mindspeed_mm.configs.config import merge_mm_args
from mindspeed_mm.configs.config import mm_extra_args_provider

# 保存mindspeed_args
mindspeed_args = get_mindspeed_args()
output_path = './recipe/mm_dance_grpo/examples/wan2.2/5B/t2v/output_args'
os.makedirs(output_path, exist_ok=True)
with open(f'{output_path}/mindspeed_args.pkl', 'wb') as f:
    pickle.dump(mindspeed_args, f)

# 保存mm args
extra_args_provider = extra_args_provider_decorator(mm_extra_args_provider)
initialize_megatron(
    extra_args_provider=extra_args_provider, args_defaults={}
)
args = get_args()
merge_mm_args(args)

with open(f'{output_path}/mm_args.pkl', 'wb') as f:
    pickle.dump(args, f)

print(">>> Success get args.")

sys.exit()
