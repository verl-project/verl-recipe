import json

from verl.utils.dataset.rl_dataset import RLHFDataset


class EvalManifestDataset(RLHFDataset):
    def _read_files_and_tokenize(self) -> None:
        super()._read_files_and_tokenize()

        manifest_path = self.config.get("eval_manifest_path")
        if not manifest_path:
            raise ValueError("data.eval_manifest_path is required")
        with open(manifest_path) as file:
            batches = json.load(file)

        if not batches or any(len(batch) != 16 for batch in batches):
            raise ValueError("Evaluation manifest must contain one or more batches of 16 task IDs")
        batches = batches * int(self.config.get("eval_manifest_repeat", 1))
        task_ids = [str(task_id) for batch in batches for task_id in batch]
        task_indices = {str(extra_info["id"]): index for index, extra_info in enumerate(self.dataframe["extra_info"])}
        missing = sorted(set(task_ids) - task_indices.keys())
        if missing:
            raise ValueError(f"Manifest task IDs are missing from the validation dataset: {missing}")
        self.dataframe = self.dataframe.select([task_indices[task_id] for task_id in task_ids])

    def __getitem__(self, index):
        row = super().__getitem__(index)
        row["extra_info"] = {
            **row["extra_info"],
            "eval_batch_index": index // 16,
            "eval_prompt_index": index % 16,
        }
        return row
