# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

import argparse
import json
import os

import datasets


def datasets_json_to_parquet(json_path):
    dataset = datasets.load_dataset("json", data_files={
        "train": json_path,
        "test": json_path
    })
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    print(f"{type(train_dataset)} {train_dataset=} ")
    print(f"{type(test_dataset)} {test_dataset=} ")

    # 只取前num_samples条数据
    num_samples = 0
    if num_samples > 0:
        train_dataset = train_dataset.select(range(min(num_samples, len(train_dataset))))
        test_dataset = test_dataset.select(range(min(num_samples, len(test_dataset))))

    def make_map_fn_to_zj2(split, json_file):
        def process_fn(example, idx):
            problem = example.pop("prompt")
            prompt = problem

            data = {
                "data_source": json_file,
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "ability": "math",
                "edit_prompt": problem,
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "question": problem,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn_to_zj2("train", json_path), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn_to_zj2("test", json_path), with_indices=True)

    local_dir = args.local_dir
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))
    print(f"local_dir:{local_dir}")


def txt_to_json(txt_file_path, json_file_path):
    prompt_list = []

    try:
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line_num, line in enumerate(lines, 1):
                clean_line = line.strip()
                if not clean_line:
                    continue
                prompt_list.append({"prompt": clean_line})

        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(prompt_list, f, indent=4, ensure_ascii=False)
        print(f"Conversion completed! {json_file_path} generated with {len(prompt_list)} valid prompts")

    except FileNotFoundError:
        print(f"Error: File {txt_file_path} not found, please check the file path")
    except Exception as e:
        print(f"Conversion failed: {str(e)}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Current_dir: {current_dir}")
    data_text = current_dir + '/../data/prompt.txt'
    data_json = current_dir + '/../data/prompt.json'
    txt_to_json(data_text, data_json)
    data_save = current_dir + '/../data/parquet/'

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=data_save)
    args = parser.parse_args()

    datasets_json_to_parquet(data_json)
