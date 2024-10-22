## Uncovering Modularity in Developmentally Plausible LMs
This repository contains code for the paper **Developmentally Plausible Multimodal Language Models Are Highly Modular**, submitted to the 2nd BabyLM Shared task (CoNLL 2024).

### Contents
1) **Benchmarking:** evaluation of pretrained multimodal language models (supported architectures: GIT, Flamingo) on common text-only and multimodal benchmarks (BLiMP, VQA, Winoground, MMStar, EWoK)
2) **Top Neuron Analysis:** identification of most important MLP neurons per task using attribution patching
3) **Neuron Ablation:** for verification

Our pretrained models submitted to the BabyLM Challenge can be found here:

- [babylm2024-multimodal](https://huggingface.co/AlinaKl/babylm2024-git-vision)
- [babylm2024-text-only](https://huggingface.co/AlinaKl/babylm2024-git-txt)

For experiments using the Flamingo architecture, the official BabyLM baseline model, pretrained on the BabyLM multimodal corpus is used:

- [babylm2024-flamingo](https://huggingface.co/babylm/flamingo-2024)


## Prerequisites
- **Data Download:** download evaluation benchmarks and place them into a folder named ``data/`` (see [BabyLM Evaluation Pipeline](https://github.com/babylm/evaluation-pipeline-2024) for detailed downloading steps)
- **Dependencies:** see [requirements.txt](requirements.txt)

## Content

### 1) Benchmarking
- `evaluate_tasks.py`: 
    - evaluate accuracy of trained model on BabyLM benchmark datasets (BLiMP, VQA, Winoground, and, additionally MMStar) 
    - without use of the BabyLM evaluation harness for faster and correct inference
    - writes results to console
    - allows exclusion of visual information
    - allows noise instead of visual information 
    - **Execution:** ``python evaluate_tasks.py`` -> benchmark dataset and evaluated model are specified in main method

### 2) Top Neuron Analysis
- `get_top_neurons.py`: 
    - compute mean activations of each MLP 
    use attribution patching with mean ablation to extract most important neurons per subtask of each benchmark dataset
    - **Execution Flamingo:** ``python get_top_neurons.py <config_file.json> flamingo``
    - **Execution GIT:** ``python get_top_neurons.py <config_file.json>``
    - config files in ``configs/`` contain information on benchmark dataset, GIT model (if GIT is analysed), padding length, etc.

- `loading_utils.py`: helper script to load datasets for attribution patching
- `plot_top_neurons.iypnb`: plot heatmaps of shared top neurons across benchmarks
- `data/top_neurons/`: results as pickle files

### 3) Neuron Ablation
- `top_neuron_ablation_vqa.py`: 
    - replace top neurons per VQA subtask with mean activations across dataset and evaluate clean vs ablated performance
    - provide config name as additional command line argument to specify hyperparameters
- `analyse_ablations.ipynb`: plot patching effect of ablating top neurons with their mean activation
- `data/ablation_top_neurons/vqa/`: results as pickle files

### Subtask Categories
Mapping between original question types and merged subtask categories:

- `vqa_superclasses.txt`
- `winoground_superclasses.txt`

Subtask Abbreviations for all Benchmarks

- `subtask_abbr.txt`