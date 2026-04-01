# Reinforcement learning for qwen3.5 models using MindSpeed-mm as the backend
<p align="center">

## 1. Environment installation ##

\[You are advised to use the matching environment version during model development.\]

For details, see [Installation Guide](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/zh/pytorch/installation.md).

```shell
# Importing CANN Environment Variables
#source /usr/local/Ascend/ascend-toolkit/set_env.sh
#source /usr/local/Ascend/nnal/atb/set_env.sh
# CANN 9.0.0.B030

# python3.11
conda create -n test python=3.11
conda activate test

# install vllm
pip install vllm==0.18.0

# install vllm-ascend
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git checkout 54879467c41784a446aa5b486a391d9bfbf488fa
pip install -r requirements.txt
export COMPILE_CUSTOM_KERNELS=1
pip install -v -e .
cd ..


# install verl
git clone https://github.com/volcengine/verl.git
cd verl
git checkout 4045d67063052dcb800c918c107b8d5a87046006
pip install -e .
cd ..

# Update the recipe directory.
git clone https://github.com/verl-project/verl-recipe.git
mkdir verl/recipe/grpo_mindspeed_mm
cp -rf verl-recipe/grpo_mindspeed_mm verl/recipe/

# install MindSpeed-MM
git clone https://gitcode.com/Ascend/MindSpeed-MM.git
cd MindSpeed-MM
bash scripts/install.sh --msid eb10b92 && bash examples/qwen3_5/install_extensions.sh
# torch version mismatch detected. Reinstall PyTorch? (y/n) -> n
# Reinstall torch_npu to match PyTorch version? (y/n) -> n
cd ..

# install transformers
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout cc7ab9be508ce6ed3637bba9e50367b29b742dc6
pip install -e .
cd ..

pip install torch_npu==2.9.0 torchvision==0.24.0  torchaudio==2.9.0 accelerate==1.13.0
pip install mathruler

# The directory structure after the preparation is as follows:
# MindSpeed-MM
# verl-recipe
# verl
# ├── recipe
#     ├── grpo_mindspeed_mm
# vllm-ascend
```

## 2. Training model preparation ##

Qwen3.5 27B model download address:

https://huggingface.co/Qwen/Qwen3.5-27B

The downloaded model is in the huggingface format and needs to be converted to the dcp format for training. For details, see the following section. 

### convert HF weight to DCP weight ###

Run the following script to convert the weight:

```shell
cd MindSpeed-MM

mm-convert Qwen35Converter hf_to_dcp \
--hf_dir ckpt/hf_path/xxxxxxx \
--dcp_dir ckpt/dcp_path/xxxxxxx

# Structure
# ———— xxxxxxx
#   |—— release
#   |—— latest_checkpointed_iteration.txt
```


Parameters in the weight conversion script are described as follows:

| Parameters        | Meaning:                                                  |
| ----------------- | --------------------------------------------------------- |
| --hf_dir | Original weight path of the huggingface                                      |
| --dcp_dir | Path for storing weights after conversion or segmentation |


## 3. Parameters for configuring args ##

Modify the following parameters and run the script to generate the args file for training preparation:

| Configuration File                                                   | Modifying a field  | Modification Description                                                  |
|----------------------------------------------------------------------|--------------------|---------------------------------------------------------------------------|
| verl/recipe/grpo_mindspeed_mm/examples/qwen3_5_27B_config.yaml | model_name_or_path | Huggingface weight path                                                   |
| verl/recipe/grpo_mindspeed_mm/examples/qwen3_5_27B_config.yaml | load               | DCP weight path                                                           |
| verl/recipe/grpo_mindspeed_mm/run_qwen3_5-27b_npu.sh    | MODEL_PATH          | Huggingface weight path |
| verl/recipe/grpo_mindspeed_mm/run_qwen3_5-27b_npu.sh       | TRAIN_FILE      | dataset for train                                                         |
| verl/recipe/grpo_mindspeed_mm/run_qwen3_5-27b_npu.sh       | TEST_FILE      | dataset for test                                                          |


```shell
# source /usr/local/Ascend/cann/set_env.sh
# cd verl
bash recipe/grpo_mindspeed_mm/run_qwen3_5-27b_npu.sh
```
