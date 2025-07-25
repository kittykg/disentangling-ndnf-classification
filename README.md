# Disentangling Neural Disjunctive Normal Form Models

This repository contains the methods and experiments for the paper
"Disentangling Neural Disjunctive Normal Form Models" at [NeSy
2025](https://2025.nesyconf.org/). The arxiv pre-print is available at
https://arxiv.org/abs/2507.10546.

We identify that performance degradation in the post-training symbolic
translation process of a neural-DNF based model [CR21, BR23, BDR25] is caused by
the thresholding discretisation method. The thresholding method is unable to
disentangle learned knowledge represented in the form of the networks' weights.
We address this issue by proposing a new disentanglement method, thereby better
preserving the models' performance.

## Dependencies

The main dependencies are:

- [neural-dnf](https://github.com/kittykg/neural-dnf) >= 2.0.0 (The
  post-training with disentanglement is available from version 2.0.0 onwards)
- [hydra](https://hydra.cc/)
- [pytorch](https://pytorch.org/)
- [wandb](https://wandb.ai/)
- [ucimlrepo](https://github.com/uci-ml-repo/ucimlrepo)

Other libraries are in the `requirements.txt` file.

## Running Instructions

### Configuration Files

We use Hydra to manage configurations. We refer new users to the [Hydra
documentation](https://hydra.cc/docs/intro/) for more information on how to use
Hydra. The configuration files are located in `conf/` . The main configuration
file is `conf/config.yaml`.

We use [Weight and Biases](https://wandb.ai/site) for logging. We refer new
users to the [WandB documentation](https://docs.wandb.ai/) for more information
on how to use WandB. To enable logging, you need to set up your own project and
API key. To disable wandb logging, set the hydra config `wandb.use_wandb` to
`False`.

We also support Discord webhook for notification. To disable it, set the hydra
config `webhook.use_discord_webhook` to `False`.

Most training configurations in `conf/training` are the final hyperparameters we
used in the paper.

### Structure

We put each dataset's scripts in their own directory. Each directory contains
the following files:

- `data_preprocessing_<dataset_name>.py`: the data-preprocessing script
  (optional)
- `train_<dataset_name>.py`: the training script
- `models.py`: the model definition
- `data_utils_<dataset_name>.py`: the data-loading functions
- `eval`: evaluation scripts, see later section

To run the data-preprocessing script, run:

```bash
python <dataset_name>/data_preprocessing_<dataset_name>.py
```

The data-preprocessing relevant configurations are in `conf/dataset`.

To run the training script, run:

```bash
python <dataset_name>/train_<dataset_name>.py \
    training=<training_config_name> \
    dataset=<dataset_name>
```

The training relevant configurations are in `conf/training`.

### Evaluation

The `eval/` directory in each dataset's directory contains the evaluation
scripts.

Key evaluation scripts are:

-  `..._after_train_eval.py`: the evaluation script after training (for MLP +
   neural DNF-based models)
- `ndnf_..._prune.py`: the pruning script for neural DNF-based models
- `ndnf_..._threshold.py`: the thresholding script for neural DNF-based models
- `ndnf_..._disentangle(_v2|_v3).py`: the disentanglement script for neural
  DNF-based models
    - Disentanglement is done on the disjunctive layer.
    - v1: thresholding the disjunctive layer with value 0
    - v2: full sweep of threshold values
    - v3 (not always available): disentangle the disjunctive layer
- `ndnf_..._asp_translation.py`: the ASP translation script for neural DNF-based
  models
- `ndnf_mt_..._soft_extraction.py`: the soft extraction script for neural DNF-MT
  models
    - Disentangle/threshold the conjunctive layer, and the disjunctive remains
      the same to output probabilities
- `ndnf_mt_..._conj_asp_translation_stats.py`: the ASP translation statistics
  script for neural DNF-MT models's conjunctive layer

To run the evaluation script, run:

```bash
python <dataset_name>/eval/<evaluation_script_name>.py \
    +eval=<evaluation_config_name> \
    ++eval.storage_dir=<storage_dir>
```

The evaluation relevant configurations are in `conf/eval`.

## References

[CR21] Cingillioglu, N., & Russo, A. (2021). pix2rule: End-to-end Neuro-symbolic
Rule Learning. In A. S. D. Garcez & E. Jiménez-Ruiz (Eds.), Proceedings of the
15th International Workshop on Neural-Symbolic Learning and Reasoning as part of
the 1st International Joint Conference on Learning & Reasoning (IJCLR 2021),
Virtual conference, October 25-27, 2021 (pp. 15–56). Retrieved from
https://ceur-ws.org/Vol-2986/paper3.pdf

[BR23] Baugh, K. G., Cingillioglu, N., & Russo, A. (2023). Neuro-symbolic Rule
Learning in Real-world Classification Tasks. In A. Martin, H.-G. Fill, A.
Gerber, K. Hinkelmann, D. Lenat, R. Stolle, & F. van Harmelen (Eds.),
Proceedings of the AAAI 2023 Spring Symposium on Challenges Requiring the
Combination of Machine Learning and Knowledge Engineering (AAAI-MAKE 2023),
Hyatt Regency, San Francisco Airport, California, USA, March 27-29, 2023.
Retrieved from https://ceur-ws.org/Vol-3433/paper12.pdf

[BDR25] Baugh, K. G., Dickens, L., & Russo, A. (2025). Neural DNF-MT: A
Neuro-symbolic Approach for Learning Interpretable and Editable Policies. In
Proc. of the 24th International Conference on Autonomous Agents and Multiagent
Systems (AAMAS 2025), Detroit, Michigan, USA, May 19 – 23, 2025, IFAAMAS.
https://dl.acm.org/doi/10.5555/3709347.3743538
