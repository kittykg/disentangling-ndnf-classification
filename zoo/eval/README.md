# Eval script for Zoo dataset

This folder contains the evaluation script for the Zoo dataset. Since the Zoo
dataset is very small, we do not have a dedicated holdout test set. Instead, we
do a 4-fold stratified cross-validation. The split uses the same seed as the
one used for the training.

The relevant configs are in the `conf/eval/` folder.

We provide a template of shell script with all commands needed in
`eval_template_sh` file. You can copy this file and modify it to your needs.

## Structure

### General files

`asp_eval_common.py`

`ndnf_eval_common.py`

### Evaluation scripts

(1) Pruning of the neural DNF-EO model -- `ndnf_eo_kfold_prune.py`

(2) Discretisation of the neural DNF-EO model

We provide two different method to discretise the neural DNF-EO model.

(2.a) Thresholding: `ndnf_eo_kfold_threshold.py`

(2.b) Disentangling: `ndnf_eo_kfold_disentangle.py` and
`ndnf_eo_kfold_disentangle_v2.py`

(3) ASP translation (with evaluation) -- `ndnf_kfold_asp_translation.py`
