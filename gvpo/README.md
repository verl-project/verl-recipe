<div align="center">

# GVPO: Group Variance Policy Optimization

[![NeurIPS](https://img.shields.io/badge/NeurIPS-b693f9?style=for-the-badge&logo=neurips&logoColor=white)](https://neurips.cc/virtual/2025/poster/117119)
[![Arxiv](https://img.shields.io/badge/Arxiv-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2504.19599)
[![GitHub](https://img.shields.io/badge/Code-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/jszkc/GVPO)
[![机器之心](https://img.shields.io/badge/机器之心-07C160?style=for-the-badge&logo=wechat&logoColor=white)](https://mp.weixin.qq.com/s/mhv0bo0PEB67jbUkZU3sXg)
[![知乎](https://img.shields.io/badge/知乎-0084FF?style=for-the-badge&logo=zhihu&logoColor=white)](https://zhuanlan.zhihu.com/p/1911487456173359632)

</div>


## ⭐️ Overview

**GVPO (Group Variance Policy Optimization)** is a **reinforcement learning algorithm** designed for **post-training large language models (LLMs)**. It provides both a theoretically sound and practically useful advancement for optimizing policies.

### 🚀 Key Highlights

- **Unique Optimal Solution:**
  GVPO guarantees convergence to a unique solution that maximizes the following objective:

  $$max_{\pi_{\theta}} \mathbb{E}_{x\sim\mathcal{D},y\sim\pi_\theta(y|x)}[R(x,y)]-\beta\mathbb{D}_{KL}[\pi_\theta(y|x)||\pi_{\theta^\prime}(y|x)]$$

- **No Importance Sampling:**
  Improves stability by eliminating the need for importance weighting.

- **Off-Policy Flexibility:**
  Supports **diverse off-policy sampling distributions**, including **experience replay** and **human demonstrations**.


## 🧩 Getting Started

### 1. Installation

This project uses **[verl](https://verl.readthedocs.io/en/latest/start/install.html)** (v0.6.0).

To install verl, please refer to the [official installation guide](https://verl.readthedocs.io/en/latest/start/install.html).

---

### 2. Data Preparation

We follow the GRPO training setup from the official verl [example script](https://github.com/volcengine/verl/blob/ddd86f527a4af75095e4677b02b5aa272913a088/examples/grpo_trainer/run_qwen2-7b_math.sh), which uses the following datasets:

- `DigitalLearningGmbH/MATH-lighteval`
- `openai/gsm8k`

To download and preprocess these datasets, run:

```bash
python examples/data_preprocess/math_dataset.py
python examples/data_preprocess/gsm8k.py
```

---

### 3. Training

Before launching training, ensure that the **model path** in the script is correctly set.

To start GVPO training:

```bash
bash recipe/gvpo/run_qwen2-7b_math_gvpo.sh
```


## 📘 Documentation

Below we summarize the main components and logic of this GVPO implementation.


### `gvpo_core_algos.py`

This module defines the core **GVPO loss function**, formulated as a Mean Squared Error (MSE):

$$\mathcal{L}_{\text{GVPO}}(\theta)=\frac{1}{2}\sum_{x, \{y_i\} } \sum_{i=1}^k [(R_\theta(x,y_i)-\overline{R_\theta(x,\{y_i\})})-(R(x,y_i)-\overline{R(x,\{y_i\})})]^2$$

This function aggregates statistics across GPUs to compute the group-mean log ratios.

```python
def compute_policy_loss_gvpo(old_log_prob, log_prob, advantages, response_mask, beta, uid, device_mesh, n):

    rtheta = ((log_prob * response_mask).sum(dim=-1) - (old_log_prob * response_mask).sum(dim=-1)) * beta
    r_minus_avg = (advantages * response_mask).sum(dim=-1) / response_mask.sum(dim=-1)

    process_group = device_mesh._flatten().get_group()
    group_size = torch.distributed.get_world_size(group=process_group)
    data = {"rtheta": rtheta.clone().detach(), "uid": uid}
    data = allgather_dict_tensors(data,group_size,process_group)

    unique_uids = torch.unique(data['uid'])
    means = {}
    for u in unique_uids:
        mask = (data['uid'] == u)
        mean_val = data['rtheta'][mask].mean()
        assert data['rtheta'][mask].shape[0] == n
        means[u.item()] = mean_val
    pg_loss = 0
    for i in range(len(rtheta)):
        pg_loss += 0.5 * ((rtheta[i] - means[uid[i].item()]) - r_minus_avg[i])**2
    pg_loss = pg_loss / (n-1)
    return pg_loss
```

---

### `gvpo_ray_trainer.py`

We modify the `_balance_batch` method to ensure that **all responses from the same prompt group** are processed within the same training iteration.  
This grouping is necessary for correctly computing group-level log-ratio averages.

```python
def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
```
> **Note:**  
> Since each iteration processes `world_size * ppo_micro_batch_size_per_gpu` responses,  
> the number of rollouts per prompt (`rollout.n`) must evenly divide this total.

---

### `run_qwen2-7b_math_gvpo.sh`

This script adapts verl’s GRPO training configuration for GVPO.  
The following parameters are particularly important:

```bash
algorithm.adv_estimator=grpo
actor_rollout_ref.actor.use_kl_loss=False
actor_rollout_ref.actor.policy_loss.loss_mode="gvpo"
actor_rollout_ref.actor.gvpo_beta=0.1
algorithm.use_kl_in_reward=False
algorithm.norm_adv_by_std_in_grpo=False
```

When converting a GRPO training script to GVPO, make sure to **update these parameters accordingly**.

> **Note:**  
> This training script is based on verl’s official examples.  
> Some settings may differ from those described in the GVPO paper—please check carefully when reproducing results.

## ✍️ Citation

If you use GVPO in your research, please cite this paper once the corresponding paper is available:

```
@article{zhang2025gvpo,
  title={GVPO: Group variance policy optimization for large language model post-training},
  author={Zhang, Kaichen and Hong, Yuzhong and Bao, Junwei and Jiang, Hongfei and Song, Yang and Hong, Dingqian and Xiong, Hui},
  journal={arXiv preprint arXiv:2504.19599},
  year={2025}
}
```