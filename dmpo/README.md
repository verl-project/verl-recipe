# Beyond Mode Collapse: Distribution Matching for Diverse Reasoning

See [`REQUIRED_VERL.txt`](REQUIRED_VERL.txt) for the upstream repository, install mode (rolling `main`, pinned release tag, or pinned git commit), and copy-pastable `pip` / `git` instructions where they exist.


This repository hosts the community implementation for the paper [Beyond Mode Collapse: Distribution Matching for Diverse Reasoning](https://arxiv.org/pdf/2605.19461). 

DMPO adds a group-wise distribution-matching objective over rollouts that share the same prompt `uid`.

The default implementation is `grpo_dmpo`, which combines the standard GRPO policy loss with the DMPO
distribution-matching loss. The recipe also provides these variants:

- `grpo_dmpo_zero`: skips zero-advantage groups during training, so groups without a useful advantage signal do not
  contribute to the DMPO term.
- `grpo_dmpo_js`: computes the gap between the current distribution and the target distribution with
  Jensen-Shannon divergence instead of the default MSE objective.
- `pure_dmpo`: updates only with the DMPO objective and does not include the GRPO policy loss.

## Usage

Run from a verl checkout that has this repository mounted as the `recipe` submodule:

```bash
bash recipe/dmpo/run_qwen2.5-7b_math_grpo_dmpo_zero.sh
```


## 🖊️ Citation

If you find this work helpful, please consider to **star🌟** this repo and cite this paper. Thanks for your support!

```bib
@misc{li2026modecollapsedistributionmatching,
      title={Beyond Mode Collapse: Distribution Matching for Diverse Reasoning}, 
      author={Xiaozhe Li and Yang Li and Xinyu Fang and Shengyuan Ding and Peiji Li and Yongkang Chen and Yichuan Ma and Tianyi Lyu and Linyang Li and Dahua Lin and Qipeng Guo and Qingwen Liu and Kai Chen},
      year={2026},
      eprint={2605.19461},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2605.19461}, 
}
```

