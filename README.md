## Uncovering Modularity in Developmentally Plausible LMs
This repository contains code for the paper **Developmentally Plausible Multimodal Language Models Are Highly Modular**, submitted to the 2nd BabyLM Shared task (CoNLL 2024).

It enables evaluation as well as top neuron identification via attribution patching of pretrained multimodal language models.

Our pretrained models submitted to the BabyLM Challenge can be found here:

- [babylm2024-multimodal](https://huggingface.co/AlinaKl/babylm2024-git-vision)
- [babylm2024-text-only](https://huggingface.co/AlinaKl/babylm2024-git-txt)


## Prerequisites
- download evaluation benchmarks and place them into a folder named data (see [BabyLM Evaluation Pipeline](https://github.com/babylm/evaluation-pipeline-2024) for detailed downloading steps)
- for dependencies, see [requirements.txt](requirements.txt)

## Content

### Top Neuron Analysis
- `get_top_neurons.py`: 
    - use attribution patching with mean ablation to extract most important neurons per task
    - provide config name as additional command line argument to specify dataset to analyze
- `loading_utils.py`: helper script to load datasets for attribution patching
- `plot_top_neurons.iypnb`: plot heatmaps of shared top neurons across benchmarks
- `data/top_neurons/`: results as pickle files

### Neuron Ablation
- `top_neuron_ablation_vqa.py`: 
    - replace top neurons per VQA subtask with mean activations across dataset and evaluate clean vs ablated performance
    - provide config name as additional command line argument to specify hyperparameters
- `analyse_ablations.ipynb`: plot patching effect of ablating top neurons with their mean activation
- `data/ablation_top_neurons/vqa/`: results as pickle files

### Benchmarking
- `evaluate_tasks.py`: evaluate accuracy of trained model on different benchmarks outside BabyLM evaluation harness (writes results to console)

### Subtask Categories
Mapping between original question types and merged subtask categories:

- `vqa_superclasses.txt`
- `winoground_superclasses.txt`

Subtask Abbreviations for all Benchmarks

- `subtask_abbr.txt`