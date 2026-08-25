# Hyperparameter Search

Submit one Slurm job per trial in a hyperparameter-search YAML. Shared launch
values go at the top and apply to every trial.

## Automodel SFT

```bash
HYPERPARAM_CONFIG=avlm/training/hyperparam_search/configs/automodel_sft/sft_hyperparam_search_trials.yaml \
  bash avlm/training/hyperparam_search/automodel_sft_sbatch_starter.sh

HYPERPARAM_CONFIG=avlm/training/hyperparam_search/configs/automodel_sft/sft_hyperparam_search_trials.yaml \
  num_nodes=1 MAX_STEPS=10 \
  bash avlm/training/hyperparam_search/automodel_sft_sbatch_starter.sh
```

## Automodel LoRA

```bash
HYPERPARAM_CONFIG=avlm/training/hyperparam_search/configs/automodel_lora/lora_hyperparam_search_trials.yaml \
  bash avlm/training/hyperparam_search/automodel_lora_sbatch_starter.sh

HYPERPARAM_CONFIG=avlm/training/hyperparam_search/configs/automodel_lora/lora_hyperparam_search_trials.yaml \
  num_nodes=1 MAX_STEPS=10 \
  bash avlm/training/hyperparam_search/automodel_lora_sbatch_starter.sh
```

## Megatron-Bridge SFT

```bash
HYPERPARAM_CONFIG=avlm/training/hyperparam_search/configs/mbridge_sft/mbridge_sft_hyperparam_search_trials.yaml \
  bash avlm/training/hyperparam_search/mbridge_sft_sbatch_starter.sh

HYPERPARAM_CONFIG=avlm/training/hyperparam_search/configs/mbridge_sft/mbridge_sft_hyperparam_search_trials.yaml \
  num_nodes=2 MAX_STEPS=10 \
  bash avlm/training/hyperparam_search/mbridge_sft_sbatch_starter.sh
```

## Megatron-Bridge LoRA

```bash
HYPERPARAM_CONFIG=avlm/training/hyperparam_search/configs/mbridge_lora/mbridge_lora_hyperparam_search_trials.yaml \
  bash avlm/training/hyperparam_search/mbridge_lora_sbatch_starter.sh

HYPERPARAM_CONFIG=avlm/training/hyperparam_search/configs/mbridge_lora/mbridge_lora_hyperparam_search_trials.yaml \
  num_nodes=2 MAX_STEPS=10 \
  bash avlm/training/hyperparam_search/mbridge_lora_sbatch_starter.sh
```
