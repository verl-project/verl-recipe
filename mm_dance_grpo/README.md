<p align="center">
<h1 align="center"> Recipe: Reinforcement learning for generative models using MindSpeed-mm as the backend (DanceGRPO) </h1>

## 1. Environment installation ##

\[You are advised to use the matching environment version during model development.\]

For details, see[Installation Guide](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/user-guide/installation.md).

```shell
# Importing CAN Environment Variables
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# Creating the python3.11 conda environment
conda create -n test python=3.11
conda activate test

# Download and Install Verl
git clone https://github.com/volcengine/verl.git
cd verl
git checkout v0.7.0
pip install -r requirements-npu.txt
pip install -r requirements.txt
pip install -v -e .
cd ..

# Update the recipe directory.
git clone https://github.com/verl-project/verl-recipe.git
cp -rf verl-recipe/mm_dance_grpo verl/recipe/

# Installing the Mindspeed-MM
git clone https://gitcode.com/Ascend/MindSpeed-MM.git
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_v0.12.1
cp -r megatron ../MindSpeed-MM/
cd ../MindSpeed-MM

# Installing the Acceleration Library
git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed
# checkout commit from MindSpeed core_r0.12.1
git checkout 93c45456c7044bacddebc5072316c01006c938f9
pip install -e .
cd ..

# Install the other dependent libraries.
pip install -e .
cd ..

# Installing the HPSv3 Scoring Model
git clone https://github.com/MizzenAI/HPSv3.git
cd HPSv3
git checkout upgrade_transformers_version
pip install -e .
cd ..

# Installing Other Packages
pip install diffusers==0.35.1 peft==0.17.1 torch_npu==2.7.1 loguru==0.7.3 opencv-python-headless==4.10.0.84 tf-keras matplotlib==3.8.4

cd verl

# Creating a Soft Link
ln -s ../MindSpeed-MM/megatron/ ./megatron
ln -s ../MindSpeed-MM/mindspeed_mm/ ./mindspeed_mm


# The directory structure after the preparation is as follows:
# HPSv3
# Megatron-LM
# MindSpeed-MM
# verl-recipe
# verl
# ├── recipe
#     ├── mm_dance_grpo
```

## 2. Dataset preparation ##

Reference ` verl/recipe/mm_dance_grpo/data/prompt.txt ` In the example provided in, you can replace the customized prompt text and run the following command to generate a parquet file:

```shell
# cd verl

python ./recipe/mm_dance_grpo/dataset/data_process.py

# Check whether the parquet file is generated in ./recipe/mm_dance_grpo/data/parquet.
```

## 3. Training model preparation ##

Wan2.2 5B model download address:

https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers

The downloaded model is in the huggingface format and needs to be converted to the dcp format for training. For details, see the following section. If you need to convert the model back to the huggingface format after training, see the section about converting the dcp format to hf.

### 3.1 HF to DCP ###

1.  Weight of the downloaded Wan2.2 model`transformer`In the mm root directory, run the following script to convert the weight:

```shell
cd MindSpeed-MM

mm-convert WanConverter hf_to_mm \
 --cfg.source_path ./weights/Wan-AI/Wan2.2-TI2V-5B-Diffusers/transformer \
 --cfg.target_path ./mm_weights/Wan-AI/Wan2.2-TI2V-5B-Diffusers/transformer
```

2.  After the weights are further converted to the DCP format, the ckpt is loaded in distributed mode during startup, which reduces the peak memory pressure on the host. The conversion command is as follows:

```shell
mm-convert WanConverter mm_to_dcp \
 --cfg.source_path ./mm_weights/Wan-AI/Wan2.2-TI2V-5B-Diffusers/transformer \
 --cfg.target_path ./dcp_weights/Wan-AI/Wan2.2-TI2V-5B-Diffusers/transformer
```

### 3.2 DCP conversion to HF ###

Currently, the format for saving the mm backend training is dcp. To convert the format to huggingface, perform the following operations:

1. Convert the weight in the dcp format to the mm weight.

```shell
cd MindSpeed-MM

# Replace the weight path saved after the training is complete.
save_path="./wandit_weight_save"
iter_dir="$save_path/iter_$(printf "%07d" $(cat $save_path/latest_checkpointed_iteration.txt))"
# Target path for weight conversion
convert_dir="./dcp_to_torch"
mkdir -p $convert_dir/release/mp_rank_00
cp $save_path/latest_checkpointed_iteration.txt $convert_dir/
echo "release" > $convert_dir/latest_checkpointed_iteration.txt
python -m torch.distributed.checkpoint.format_utils dcp_to_torch "$iter_dir" "$convert_dir/release/mp_rank_00/model_optim_rng.pt"
```

2. Run the following command in the root directory to convert the mm weight to the hf weight:

```shell
mm-convert WanConverter mm_to_hf \
 --cfg.source_path path_for_your_saved_weight \
 --cfg.target_path ./converted_weights/Wan-AI/Wan2.2-TI2V-5B-Diffusers/transformer \
 --cfg.hf_dir weights/Wan-AI/Wan2.2-TI2V-5B-Diffusers/transformer
```

Parameters in the weight conversion script are described as follows:

| Parameters        | Meaning:                                                  |
| ----------------- | --------------------------------------------------------- |
| --cfg.source_path | Original Weight Path                                      |
| --cfg.target_path | Path for storing weights after conversion or segmentation |
| --cfg.hf_dir      | Original weight path of the huggingface                   |

## 4. Scoring Model preparation ##

1. Download the HPSv3 model: https://huggingface.co/MizzenAI/HPSv3/tree/main
In the training script, change the value of +actor_rollout_ref.model.reward_model_path to the HPSv3 weight path.
```shell
+actor_rollout_ref.model.reward_model_path=/home/CKPT/HPSv3/HPSv3.safetensors
```

2. Download the Qwen2.5VL 7B model: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/tree/main
Modify the HPSv3/hpsv3/config/HPSv3_7B.yaml file to configure the directory for the Qwen2-VL-7B model.
```shell
model_name_or_path: "/home/CKPT/Qwen2-VL-7B-Instruct"
```

## 5. Parameters for configuring args ##

Modify the following parameters and run the script to generate the args file for training preparation:

| Configuration File                                                   | Modifying a field | Modification Description                                                                                           |
|----------------------------------------------------------------------| ----------------- | ------------------------------------------------------------------------------------------------------------------ |
| verl/recipe/mm_dance_grpo/examples/wan2.2/5B/t2v/model.json | from_pretrained   | Set this parameter to the path corresponding to the downloaded weight (including tokenizer, ae, and text_encoder). |
| verl/recipe/mm_dance_grpo/examples/wan2.2/5B/t2v/get_train_args.sh    | LOAD_PATH         | Pre-training weight path after DCP weight conversion in the WAN 2.2 model                                          |
| verl/recipe/mm_dance_grpo/examples/wan2.2/5B/fsdp2_config.yaml       | sharding_size     | Indicates the number of fragments with the weight in FSDP2 mode, which is usually the same as the number of cards. |

```shell
# source /usr/local/Ascend/cann/set_env.sh
# cd verl

bash ./recipe/mm_dance_grpo/examples/wan2.2/5B/t2v/get_train_args.sh

# Check whether the mindspeed_args.pkl and mm_args.pkl files are generated in ./recipe/mm_dance_grpo/examples/wan2.2/5B/t2v/output_args.
```

## 6. Model RL training ##

Modifying Parameters in the verl/recipe/mm_dance_grpo/run_verl_dance.sh RL Training Script

| Field Name                      | Modification Description                                                                                                                                  |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| reward_model_path               | Reward model address, which needs to be provided in HPSv3.safetensors.                                                                                    |
| train_data / test_data          | Generated dataset address                                                                                                                                 |
| model_args_path                 | Path for saving the generated model arguments.                                                                                                            |
| data.train_batch_size           | RL-trained GBS                                                                                                                                            |
| +actor_rollout_ref.rollout.only | Indicates whether to enable the only inference function. The inference result will be saved in the +actor_rollout_ref.rollout.result.save.path directory. |

Start RL training.

```shell
# cd verl

bash recipe/mm_dance_grpo/run_verl_dance.sh
```
