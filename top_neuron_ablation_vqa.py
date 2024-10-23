import torch
import torch.nn.functional as F

import gc
import sys
import os
import numpy as np
from tqdm import tqdm
import pickle
import json
from collections import defaultdict
import random
from datasets import load_dataset

from loading_utils import (
    parse_vqa_qtypes,
    load_git_model,
    create_nnsight,
    load_flamingo_model,
)
from get_top_neurons import extract_submodules_git, extract_submodules_flamingo


def load_subtask(task_file, top_k):
    with open(f"{task_file}", "rb") as f:
        layer_dict = pickle.load(f)
        top_neurons = []
        for layer, tensors in layer_dict.items():
            top, effects = tensors
            top_new = [f"{layer}_{neuron}" for neuron in top]
            neuron_with_effect = [
                (neuron, effect) for neuron, effect in zip(top_new, effects)
            ]
            top_neurons.extend(neuron_with_effect)
        top_neurons.sort(key=lambda x: x[1], reverse=True)
        top_neurons_dict = {}
        for n in top_neurons[:top_k]:
            neuron_name = n[0]
            parts = neuron_name.split("_")
            layer = int(parts[1])
            number = int(parts[2])
            if layer in top_neurons_dict:
                top_neurons_dict[layer].append(number)
            else:
                top_neurons_dict[layer] = [number]

        return top_neurons_dict


def patch_top_neurons(
    clean,
    img_inputs,
    model,
    top_neurons_mean,
    metric_fn,
    acc_fn,
    metric_kwargs=dict(),
):
    tracer_kwargs = {"validate": False, "scan": False}

    # clean run -> model can be approximated through linear function of its activations
    if img_inputs is None:
        with model.trace(clean, **tracer_kwargs), torch.no_grad():
            metric_clean = metric_fn(model, **metric_kwargs).save()
            acc_clean = acc_fn(model, **metric_kwargs).save()

        with model.trace(clean, **tracer_kwargs), torch.no_grad():
            for layer_id, neurons in top_neurons_mean.items():
                for neuron_id in neurons:
                    model.git.encoder.layer[layer_id].intermediate[0].output[
                        :, :, neuron_id
                    ] = top_neurons_mean[layer_id][neuron_id]
            metric_ablated = metric_fn(model, **metric_kwargs).save()
            acc_ablated = acc_fn(model, **metric_kwargs).save()

    else:
        with model.trace(
            clean, pixel_values=img_inputs, **tracer_kwargs
        ), torch.no_grad():
            metric_clean = metric_fn(model, **metric_kwargs).save()
            acc_clean = acc_fn(model, **metric_kwargs).save()

        with model.trace(
            clean, pixel_values=img_inputs, **tracer_kwargs
        ), torch.no_grad():
            for layer_id, neurons in top_neurons_mean.items():
                for neuron_id in neurons:
                    model.git.encoder.layer[layer_id].intermediate[0].output[
                        :, :, neuron_id
                    ] = top_neurons_mean[layer_id][neuron_id]
            metric_ablated = metric_fn(model, **metric_kwargs).save()
            acc_ablated = acc_fn(model, **metric_kwargs).save()

    total_effect = (metric_ablated.value - metric_clean.value).detach()

    return total_effect, acc_clean, acc_ablated


def compute_metric_with_ablation(
    nnsight_model, examples, batch_size, top_neurons_mean, noimg, flamingo=False
):
    num_examples = len(examples)
    batches = [
        examples[i : min(i + batch_size, num_examples)]
        for i in range(0, num_examples, batch_size)
    ]
    device = "cuda"

    sum_effects = None
    acc_clean_sum = None
    acc_ablated_sum = None

    if noimg:
        print("computing performance WITHOUT vision")
    else:
        print("computing performance WITH vision")

    print("ablating the following neurons:")
    for layer, edict in top_neurons_mean.items():
        print(layer, edict.keys())
    for batch in tqdm(batches):
        clean_inputs = torch.cat([e["clean_prefix"] for e in batch], dim=0).to(device)

        clean_answer_idxs = torch.tensor(
            [e["clean_answer"] for e in batch], dtype=torch.long, device=device
        )
        if noimg:
            img_inputs = None
        else:
            img_inputs = torch.cat([e["pixel_values"] for e in batch], dim=0).to(device)

        first_distractor_idxs = torch.tensor(
            [e["distractors"][0] for e in batch], dtype=torch.long, device=device
        )

        def metric(nnsight_model, flamingo=flamingo):
            # compute difference between correct answer and first distractor
            # TODO: compute avg difference between correct answer and each distractor
            # embds_out = nnsight_model.output.output.save()

            if flamingo:
                return (
                    torch.gather(
                        nnsight_model.output.logits[:, -1, :],
                        dim=-1,
                        index=first_distractor_idxs.view(-1, 1),
                    ).squeeze(-1)
                    - torch.gather(
                        nnsight_model.output.logits[:, -1, :],
                        dim=-1,
                        index=clean_answer_idxs.view(-1, 1),
                    ).squeeze(-1)
                )
            else:
                return (
                    torch.gather(
                        nnsight_model.output.output[:, -1, :],
                        dim=-1,
                        index=first_distractor_idxs.view(-1, 1),
                    ).squeeze(-1)
                    - torch.gather(
                        nnsight_model.output.output[:, -1, :],
                        dim=-1,
                        index=clean_answer_idxs.view(-1, 1),
                    ).squeeze(-1)
                )

        def get_acc(nnsight_model):

            if flamingo:
                return (
                    torch.gather(
                        nnsight_model.output.logits[:, -1, :],
                        dim=-1,
                        index=clean_answer_idxs.view(-1, 1),
                    ).squeeze(-1)
                    > torch.gather(
                        nnsight_model.output.output[:, -1, :],
                        dim=-1,
                        index=first_distractor_idxs.view(-1, 1),
                    ).squeeze(-1)
                ).int()
            else:
                return (
                    torch.gather(
                        nnsight_model.output.output[:, -1, :],
                        dim=-1,
                        index=clean_answer_idxs.view(-1, 1),
                    ).squeeze(-1)
                    > torch.gather(
                        nnsight_model.output.output[:, -1, :],
                        dim=-1,
                        index=first_distractor_idxs.view(-1, 1),
                    ).squeeze(-1)
                ).int()

        total_effect, acc_clean, acc_ablated = patch_top_neurons(
            clean_inputs,
            img_inputs,
            nnsight_model,
            top_neurons_mean,
            metric,
            get_acc,
            metric_kwargs=dict(),
        )

        if sum_effects is None:
            sum_effects = total_effect.sum(dim=0)
            acc_clean_sum = acc_clean.sum(dim=0)
            acc_ablated_sum = acc_ablated.sum(dim=0)
        else:
            sum_effects += total_effect.sum(dim=0)
            acc_clean_sum += acc_clean.sum(dim=0)
            acc_ablated_sum += acc_ablated.sum(dim=0)

    ablation_effect = sum_effects / num_examples
    acc_clean_effect = acc_clean_sum / num_examples
    acc_ablated_effect = acc_ablated_sum / num_examples
    print(f"acc clean: {acc_clean_effect}")
    print(f"acc ablated: {acc_ablated_effect}")

    return ablation_effect, acc_clean_effect, acc_ablated_effect


def select_random_neurons(layers, neurons_per_layer, seed, num_selections=100):
    selected_neurons = defaultdict(list)

    random.seed(seed)

    # Generate 100 random selections of (layer, neuron)
    for _ in range(num_selections):
        layer = random.randint(0, layers - 1)  # Random layer between 0 and layers-1
        neuron = random.randint(
            0, neurons_per_layer - 1
        )  # Random neuron between 0 and neurons_per_layer-1
        selected_neurons[layer].append(neuron)

    return dict(selected_neurons)


def get_mean_for_selected_neurons(selected_neurons, mean_act_files):
    neurons_mean = {}

    for layer, neurons in selected_neurons.items():
        if layer > len(mean_act_files) - 1:
            print(f"skipping ablation in layer: {layer}")
            continue
        neurons_mean[layer] = {}
        mean_state = np.load(mean_act_files[layer], allow_pickle=True)
        mean_state = torch.tensor(mean_state).to("cuda")
        for neuron in neurons:
            # TBD: eventuell hat mean state nicht nur n_dim dimensionen
            neurons_mean[layer][neuron] = mean_state[neuron]
        mean_state = None
        del mean_state
        torch.cuda.empty_cache()
        gc.collect()
    return neurons_mean


def load_subtask_examples(subtask, lookup, tokenizer):
    pad_to_length = 32
    hf_path = "HuggingFaceM4/VQAv2"
    hf_split = "validation"
    local_file = "data/vqa_filtered/vqa_distractors_info.json"

    # load Hugging Face dataset with streaming to avoid memory overload
    hf_ds = load_dataset(hf_path, split=hf_split)
    distractor_ds = load_dataset(path="json", name=None, data_files=local_file)["train"]

    filtered_samples = []
    eos_token_id = tokenizer.convert_tokens_to_ids("</s>")
    print("filtering subtask samples")
    for sample in tqdm(distractor_ds):
        # add distractor info from json to hf dataset sample
        img_sample = hf_ds[sample["idx_in_hf_dataset"]]
        qt_fine = img_sample["question_type"]
        qt = lookup[qt_fine]
        if qt != subtask:
            continue

        if img_sample["image_id"] != sample["image_id"]:
            print(f"Sample Mismatch: {sample['image_id']} - {img_sample['image_id']}")
            continue

        img_sample["distractors"] = sample["distractors"]

        clean_tokens = tokenizer(
            img_sample["question"], return_tensors="pt", padding=False
        )
        clean_prefix = clean_tokens.input_ids
        # add answer with preceding whitespace
        clean_answer = tokenizer(
            f" {img_sample['multiple_choice_answer']}",
            return_tensors="pt",
            padding=False,
        ).input_ids
        distractors = [
            tokenizer(f" {d}", return_tensors="pt", padding=False).input_ids
            for d in img_sample["distractors"]
        ]

        # remove EOS tokens from answers
        clean_answer = clean_answer[clean_answer != eos_token_id].unsqueeze(0)
        # only keep examples where answers are single tokens
        if clean_answer.shape[1] != 1:
            continue

        # do the same for distractors
        prepared_distractors = []
        for d in distractors:
            d = d[d != eos_token_id].unsqueeze(0)
            if d.shape[1] == 1:
                prepared_distractors.append(d.item())
        if len(prepared_distractors) == 0:
            continue

        # if we specify `pad_to_length`, left-pad all inputs to a max length
        prefix_length_wo_pad = clean_prefix.shape[1]
        tokenizer.padding_side = "right"
        pad_length = pad_to_length - prefix_length_wo_pad
        if pad_length < 0:  # example too long
            continue
        # left padding: reverse, right-pad, reverse
        clean_prefix = torch.flip(
            F.pad(
                torch.flip(clean_prefix, (1,)),
                (0, pad_length),
                value=tokenizer.pad_token_id,
            ),
            (1,),
        )

        # generate image embeddings
        pixel_values = img_processor(
            images=img_sample["image"].convert(mode="RGB"), return_tensors="pt"
        )["pixel_values"]

        example_dict = {
            "clean_prefix": clean_prefix,
            "clean_answer": clean_answer.item(),
            "distractors": prepared_distractors,
            "question_type": lookup[img_sample["question_type"]],
            "prefix_length_wo_pad": prefix_length_wo_pad,
            "pixel_values": pixel_values,
        }
        filtered_samples.append(example_dict)

    return filtered_samples


if __name__ == "__main__":

    with open(f"configs/{sys.argv[1]}", "r") as f:
        config = json.load(f)

    if len(sys.argv) == 3 and sys.argv[2] == "flamingo":
        model_setting = "flamingo"
        flamingo = True
        model, img_processor, tokenizer = load_flamingo_model()
        nnsight_model = create_nnsight(model)
        submodules = extract_submodules_flamingo(nnsight_model)
        model_prefix = f"{model_setting}"
    else:
        model_setting = config["model_path"]
        epoch = config["epoch"]
        flamingo = False
        model_dir = (  # must be changed depending on where trained models are stored
            "../babylm_GIT/models_for_eval/final_models"
        )
        model, img_processor, tokenizer = load_git_model(
            model_dir, model_setting, epoch
        )
        nnsight_model = create_nnsight(model)
        submodules = extract_submodules_git(nnsight_model)
        model_prefix = f"{model_setting}_e{epoch}"

    task = config["task"]
    num_examples = config["num_examples"]
    print(f"task: {task} num_examples: {num_examples}")

    batch_size = config["batch_size"]
    pad_len = config["pad_len"]

    noimg = config["noimg"]
    threshold_subtask = config["threshold_subtask"]

    # load and prepare model
    mlps = [
        submodules[submodule] for submodule in submodules if submodule.startswith("mlp")
    ]
    print("loaded model and submodules")

    # retrieve precomputed mean activation files
    mean_act_files = []
    prefix = (
        f"{task}_{model_prefix}_n{num_examples if num_examples != -1 else 'all'}{'_noimg' if noimg else ''}"
    )
    for file in os.listdir("mean_activations/"):
        if file.startswith(prefix + "_mean_acts"):
            mean_act_files.append(f"mean_activations/{file}")
    print("loaded mean activations")

    # load and prepare data
    lookup = parse_vqa_qtypes()
    subtasks = set(lookup.values())

    all_ablation_effects = {}
    for subtask in subtasks:
        subtask_dense = subtask.replace(" ", "")
        subtask_samples = load_subtask_examples(subtask, lookup, tokenizer)

        print(f"loaded and prepared {subtask}")

        if len(subtask_samples) < 10:
            print(f"not enough samples in {subtask}")

        # for each subtask, ablated its own neurons and the neurons of its countertask without vision

        # get top neurons
        top_neurons_file = f"data/top_neurons/vqa/vqa_{model_prefix}_nall/{subtask_dense}_top_neurons.pkl"
        top_neurons = load_subtask(top_neurons_file, top_k=100)

        top_neurons_file_noimg = f"data/top_neurons/vqa/vqa_{model_prefix}_nall_noimg/{subtask_dense}_top_neurons.pkl"
        top_neurons_noimg = load_subtask(top_neurons_file_noimg, top_k=100)

        # get their top activations
        top_neurons_mean = get_mean_for_selected_neurons(top_neurons, mean_act_files)
        top_neurons_mean_noimg = get_mean_for_selected_neurons(
            top_neurons_noimg, mean_act_files
        )

        # do attribution patching with selected top neurons per subtask
        noimg_label = [True, False]
        for noimg in noimg_label:
            print(f"noimg: {noimg}")
            ablation_effect, acc_clean, acc_ablated = compute_metric_with_ablation(
                nnsight_model,
                subtask_samples,
                batch_size,
                top_neurons_mean,
                noimg,
                flamingo,
            )
            (
                ablation_effect_noimg,
                acc_clean_noimg,
                acc_ablated_noimg,
            ) = compute_metric_with_ablation(
                nnsight_model,
                subtask_samples,
                batch_size,
                top_neurons_mean_noimg,
                noimg,
                flamingo,
            )

            app = "" if noimg is False else "_noimg"
            all_ablation_effects[subtask + app] = {
                "ablation_effect": ablation_effect,
                "acc_clean": acc_clean,
                "acc_clean_noimg": acc_clean_noimg,
                "acc_ablated": acc_ablated,
                "ablation_effect_noimg": ablation_effect_noimg,
                "acc_ablated_noimg": acc_ablated_noimg,
            }
            print(all_ablation_effects)

        top_neurons_mean, top_neurons_mean_noimg = None, None
        del top_neurons_mean, top_neurons_mean_noimg
        torch.cuda.empty_cache()
        gc.collect()

        out_dir = f"results/ablation_top_neurons/vqa/{prefix}/"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(
            f"results/ablation_top_neurons/vqa/{prefix}/{subtask_dense}_top_neurons_ablated.pkl",
            "wb",
        ) as f:
            pickle.dump(all_ablation_effects, f)
