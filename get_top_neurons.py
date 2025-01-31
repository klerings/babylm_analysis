import gc
import sys
import numpy as np
import os
from tqdm import tqdm
import torch

from loading_utils import (
    load_vqa_examples,
    load_blimp_examples,
    load_winoground_examples,
    load_mmstar_examples,
    load_ewok_examples,
    load_git_model,
    load_flamingo_model,
    create_nnsight,
)
import pickle
import json


def extract_submodules_git(model):
    submodules = {}
    for idx, layer in enumerate(model.git.encoder.layer):
        submodules[f"mlp.{idx}"] = layer.intermediate  # output of first MLP
        submodules[f"attn.{idx}"] = layer.attention  # output of attention
        submodules[f"resid.{idx}"] = layer  # output of whole layer
    return submodules


def extract_submodules_flamingo(model):
    submodules = {}
    for idx, layer in enumerate(model.model.decoder.layers):
        submodules[f"mlp.{idx}"] = layer.fc1  # output of first MLP
    return submodules


def compute_mean_activations(
    examples, model, submodules, batch_size, noimg=False, file_prefix=None
):
    tracer_kwargs = {"validate": False, "scan": False}
    device = "cuda"
    num_examples = len(examples)
    batches = [
        examples[i : min(i + batch_size, num_examples)]
        for i in range(0, num_examples, batch_size)
    ]

    def extract_hidden_states(submodule):
        total_samples = 0
        # Initialize storage for cumulative activations and count of samples
        cumulative_activations = 0

        for batch in tqdm(batches):
            clean_inputs = torch.cat([e["clean_prefix"] for e in batch], dim=0).to(
                device
            )

            # clean run -> model can be approximated through linear function of its activations
            hidden_states_clean = {}
            if noimg:
                with model.trace(clean_inputs, **tracer_kwargs), torch.no_grad():
                    x = submodule.output
                    hidden_states_clean = x.save()
            else:
                img_inputs = torch.cat([e["pixel_values"] for e in batch], dim=0).to(
                    device
                )
                with model.trace(
                    clean_inputs, pixel_values=img_inputs, **tracer_kwargs
                ), torch.no_grad():
                    x = submodule.output
                    hidden_states_clean = x.save()
            hidden_states_clean = hidden_states_clean.value

            batch_size = clean_inputs.shape[0]  # Assuming shape [batch_size, ...]
            total_samples += batch_size

            cumulative_activations += (
                hidden_states_clean.sum(dim=(0, 1)).detach().cpu()
            )  # detach

            hidden_states_clean = None
            clean_inputs = None
            state = None
            x = None
            batch_size = None
            del hidden_states_clean, clean_inputs, state, x, batch_size
            if not noimg:
                img_inputs = None
                del img_inputs
            torch.cuda.empty_cache()
            gc.collect()

        # Compute mean activation by dividing the cumulative activations by the total number of samples
        mean_activations = cumulative_activations / total_samples

        cumulative_activations = None
        del cumulative_activations
        torch.cuda.empty_cache()
        gc.collect()

        return mean_activations

    mean_act_files = []
    for i, submodule in enumerate(submodules):
        submodule_acts = extract_hidden_states(submodule)
        filename = f"mean_activations/{file_prefix}_mean_acts_{i}.npy"
        np.save(filename, submodule_acts)

        submodule_acts = None
        del submodule_acts
        torch.cuda.empty_cache()
        gc.collect()

        mean_act_files.append(filename)

    return mean_act_files


# Attribution patching with integrated gradients
def _pe_ig(
    clean,
    img_inputs,
    model,
    submodules,
    mean_act_files,
    metric_fn,
    pad_len,
    steps=10,
    metric_kwargs=dict(),
):
    tracer_kwargs = {"validate": False, "scan": False}

    # clean run -> model can be approximated through linear function of its activations
    hidden_states_clean = {}
    if img_inputs is None:
        with model.trace(clean, **tracer_kwargs), torch.no_grad():
            for submodule in submodules:
                x = submodule.output
                hidden_states_clean[submodule] = x.save()
    else:
        with model.trace(
            clean, pixel_values=img_inputs, **tracer_kwargs
        ), torch.no_grad():
            for submodule in submodules:
                x = submodule.output
                hidden_states_clean[submodule] = x.save()
    hidden_states_clean = {k: v.value for k, v in hidden_states_clean.items()}

    x = None
    del x
    torch.cuda.empty_cache()
    gc.collect()

    effects = {}
    deltas = {}
    grads = {}
    for i, submodule in enumerate(submodules):
        # load mean hidden states from file
        mean_state = np.load(mean_act_files[i], allow_pickle=True)
        mean_state = torch.tensor(mean_state).to("cuda")
        mean_state.requires_grad = True

        clean_state = hidden_states_clean[submodule]
        with model.trace(**tracer_kwargs) as tracer:
            metrics = []
            fs = []
            for step in range(steps):
                alpha = step / steps
                f = (1 - alpha) * clean_state + alpha * mean_state
                f.retain_grad()
                fs.append(f)
                if img_inputs is None:
                    with tracer.invoke(clean, scan=tracer_kwargs["scan"]):
                        submodule.output = f
                        metrics.append(metric_fn(model, **metric_kwargs))
                else:
                    with tracer.invoke(
                        clean, pixel_values=img_inputs, scan=tracer_kwargs["scan"]
                    ):
                        submodule.output = f[
                            :, -pad_len:, :
                        ]  # we are only interested in the text tokens which are appended to img tokens
                        metrics.append(metric_fn(model, **metric_kwargs))
            metric = sum([m for m in metrics])
            metric.sum().backward(retain_graph=True)

        mean_grad = sum([f.grad for f in fs]) / steps
        grad = mean_grad
        delta = (
            (mean_state - clean_state).detach()
            if mean_state is not None
            else -clean_state.detach()
        )
        effect = torch.mul(grad, delta)

        effects[submodule] = effect
        deltas[submodule] = delta
        grads[submodule] = grad

        mean_state = None
        del mean_state
        torch.cuda.empty_cache()
        gc.collect()

    return (effects, deltas, grads)


def get_important_neurons(
    model,
    examples,
    batch_size,
    mlps,
    pad_len,
    mean_act_files,
    task,
    noimg,
    flamingo=False,
):
    # uses attribution patching to identify most important neurons for subtask
    num_examples = len(examples)
    batches = [
        examples[i : min(i + batch_size, num_examples)]
        for i in range(0, num_examples, batch_size)
    ]
    device = "cuda"

    sum_effects = {}

    for batch in tqdm(batches):
        clean_inputs = torch.cat([e["clean_prefix"] for e in batch], dim=0).to(device)

        if task == "vqa":
            clean_answer_idxs = torch.tensor(
                [e["clean_answer"] for e in batch], dtype=torch.long, device=device
            )
            if noimg:
                img_inputs = None
            else:
                img_inputs = torch.cat([e["pixel_values"] for e in batch], dim=0).to(
                    device
                )

            first_distractor_idxs = torch.tensor(
                [e["distractors"][0] for e in batch], dtype=torch.long, device=device
            )

            def metric(model, flamingo=flamingo):
                # compute difference between correct answer and first distractor
                # TODO: compute avg difference between correct answer and each distractor

                if flamingo:
                    return (
                        torch.gather(
                            model.output.logits[:, -1, :],
                            dim=-1,
                            index=first_distractor_idxs.view(-1, 1),
                        ).squeeze(-1)
                        - torch.gather(
                            model.output.logits[:, -1, :],
                            dim=-1,
                            index=clean_answer_idxs.view(-1, 1),
                        ).squeeze(-1)
                    )
                else:
                    return (
                        torch.gather(
                            model.output.output[:, -1, :],
                            dim=-1,
                            index=first_distractor_idxs.view(-1, 1),
                        ).squeeze(-1)
                        - torch.gather(
                            model.output.output[:, -1, :],
                            dim=-1,
                            index=clean_answer_idxs.view(-1, 1),
                        ).squeeze(-1)
                    )

            effects, _, _ = _pe_ig(
                clean_inputs,
                img_inputs,
                model,
                mlps,
                mean_act_files,
                metric,
                pad_len,
                steps=10,
                metric_kwargs=dict(),
            )

        elif task == "blimp":
            img_inputs = None
            clean_answer_idxs = torch.tensor(
                [e["clean_answer"] for e in batch], dtype=torch.long, device=device
            )
            patch_answer_idxs = torch.tensor(
                [e["patch_answer"] for e in batch], dtype=torch.long, device=device
            )

            def metric(model, flamingo=flamingo):

                if flamingo:
                    return (
                        torch.gather(
                            model.output.logits[:, -1, :],
                            dim=-1,
                            index=patch_answer_idxs.view(-1, 1),
                        ).squeeze(-1)
                        - torch.gather(
                            model.output.logits[:, -1, :],
                            dim=-1,
                            index=clean_answer_idxs.view(-1, 1),
                        ).squeeze(-1)
                    )
                else:
                    return (
                        torch.gather(
                            model.output.output[:, -1, :],
                            dim=-1,
                            index=patch_answer_idxs.view(-1, 1),
                        ).squeeze(-1)
                        - torch.gather(
                            model.output.output[:, -1, :],
                            dim=-1,
                            index=clean_answer_idxs.view(-1, 1),
                        ).squeeze(-1)
                    )

            effects, _, _ = _pe_ig(
                clean_inputs,
                img_inputs,
                model,
                mlps,
                mean_act_files,
                metric,
                pad_len,
                steps=10,
                metric_kwargs=dict(),
            )

        elif task == "winoground" or task == "mmstar":
            if noimg:
                img_inputs = None
            else:
                img_inputs = torch.cat([e["pixel_values"] for e in batch], dim=0).to(
                    device
                )

            correct_idxs = [e["correct_idx"] for e in batch]
            incorrect_idxs = [e["incorrect_idx"] for e in batch]

            def metric(model, flamingo=flamingo):
                correct_sent_logits = []
                incorrect_sent_logits = []
                for i, (idx, cf_idx) in enumerate(zip(correct_idxs, incorrect_idxs)):
                    if flamingo:
                        logits = torch.gather(
                            model.output.logits[i, :, :],
                            dim=1,
                            index=torch.tensor([idx]).to("cuda"),
                        ).squeeze(
                            -1
                        )  # [1, seq]
                        cf_logits = torch.gather(
                            model.output.logits[i, :, :],
                            dim=1,
                            index=torch.tensor([cf_idx]).to("cuda"),
                        ).squeeze(
                            -1
                        )  # [1, seq]
                    else:
                        logits = torch.gather(
                            model.output.output[i, :, :],
                            dim=1,
                            index=torch.tensor([idx]).to("cuda"),
                        ).squeeze(
                            -1
                        )  # [1, seq]
                        cf_logits = torch.gather(
                            model.output.output[i, :, :],
                            dim=1,
                            index=torch.tensor([cf_idx]).to("cuda"),
                        ).squeeze(
                            -1
                        )  # [1, seq]
                    correct_sent_logits.append(logits.sum().unsqueeze(0))
                    incorrect_sent_logits.append(cf_logits.sum().unsqueeze(0))
                correct_sent_logits = torch.cat(correct_sent_logits, dim=0)
                incorrect_sent_logits = torch.cat(incorrect_sent_logits, dim=0)
                return incorrect_sent_logits - correct_sent_logits

            effects, _, _ = _pe_ig(
                clean_inputs,
                img_inputs,
                model,
                mlps,
                mean_act_files,
                metric,
                pad_len,
                steps=10,
                metric_kwargs=dict(),
            )

        elif task == "ewok":
            img_inputs = None

            correct_idxs = [e["correct_idx"] for e in batch]
            incorrect_idxs = [e["incorrect_idx"] for e in batch]

            def metric(model, flamingo=flamingo):
                correct_sent_logits = []
                incorrect_sent_logits = []
                for i, (idx, cf_idx) in enumerate(zip(correct_idxs, incorrect_idxs)):
                    if flamingo:
                        logits = torch.gather(
                            model.output.logits[i, :, :],
                            dim=1,
                            index=torch.tensor([idx]).to("cuda"),
                        ).squeeze(
                            -1
                        )  # [1, seq]
                        cf_logits = torch.gather(
                            model.output.logits[i, :, :],
                            dim=1,
                            index=torch.tensor([cf_idx]).to("cuda"),
                        ).squeeze(
                            -1
                        )  # [1, seq]
                    else:
                        logits = torch.gather(
                            model.output.output[i, :, :],
                            dim=1,
                            index=torch.tensor([idx]).to("cuda"),
                        ).squeeze(
                            -1
                        )  # [1, seq]
                        cf_logits = torch.gather(
                            model.output.output[i, :, :],
                            dim=1,
                            index=torch.tensor([cf_idx]).to("cuda"),
                        ).squeeze(
                            -1
                        )  # [1, seq]
                    correct_sent_logits.append(logits.sum().unsqueeze(0))
                    incorrect_sent_logits.append(cf_logits.sum().unsqueeze(0))
                correct_sent_logits = torch.cat(correct_sent_logits, dim=0)
                incorrect_sent_logits = torch.cat(incorrect_sent_logits, dim=0)
                return incorrect_sent_logits - correct_sent_logits

            effects, _, _ = _pe_ig(
                clean_inputs,
                img_inputs,
                model,
                mlps,
                mean_act_files,
                metric,
                pad_len,
                steps=10,
                metric_kwargs=dict(),
            )

        else:
            print(f"{task} is not defined")
            exit()

        for submodule in mlps:
            if submodule not in sum_effects:
                sum_effects[submodule] = effects[submodule].sum(dim=1).sum(dim=0)
            else:
                sum_effects[submodule] += effects[submodule].sum(dim=1).sum(dim=0)

    # Print top 100 neurons in each submodule (ndim=3072)
    k = 100

    top_neurons = {}
    for idx, submodule in enumerate(mlps):
        sum_effects[submodule] /= num_examples
        v, i = torch.topk(
            sum_effects[submodule].flatten(), k
        )  # v=top effects, i=top indices
        top_neurons[f"mlp_{idx}"] = (i.cpu(), v.cpu())
    return top_neurons


if __name__ == "__main__":

    with open(f"configs/{sys.argv[1]}", "r") as f:
        config = json.load(f)

    if len(sys.argv) == 3 and sys.argv[2] == "flamingo":
        model_setting = "flamingo"
        epoch = None
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

    # load and prepare data
    if task == "vqa":
        examples = load_vqa_examples(
            tokenizer, img_processor, pad_to_length=pad_len, n_samples=num_examples
        )
        subtask_key = "question_type"
    elif task == "blimp":
        examples = load_blimp_examples(
            tokenizer, pad_to_length=pad_len, n_samples=num_examples
        )
        subtask_key = "linguistics_term"
    elif task == "winoground":
        examples = load_winoground_examples(
            tokenizer, img_processor, pad_to_length=pad_len, n_samples=num_examples
        )
        subtask_key = "superclass"  # "secondary_tag"
    elif task == "mmstar":
        examples = load_mmstar_examples(
            tokenizer, img_processor, pad_to_length=pad_len, n_samples=num_examples
        )
        subtask_key = "category"
    elif task == "ewok":
        examples = load_ewok_examples(
            tokenizer, pad_to_length=pad_len, n_samples=num_examples
        )
        subtask_key = "Domain"
    else:
        print(f"{task} is not implemented")
    print(f"loaded samples: {len(examples)}")

    # precompute mean activations on task or retrieve precomputed activation files
    prefix = f"{task}_{model_prefix}_n{num_examples if num_examples != -1 else 'all'}{'_noimg' if noimg else ''}"

    mean_act_files = []
    for file in os.listdir("mean_activations/"):
        if file.startswith(prefix + "_mean_acts"):
            mean_act_files.append(f"mean_activations/{file}")
    if len(mean_act_files) != len(mlps):
        mean_act_files = compute_mean_activations(
            examples,
            nnsight_model,
            mlps,
            batch_size=128,
            noimg=noimg,
            file_prefix=prefix,
        )
        print("computed mean activations")
    else:
        print("retrieved precomputed mean activations")

    # identify subtasks
    subtasks = {}
    for e in examples:
        subtask = e[subtask_key]
        substask_nw = subtask.replace(" ", "")
        if substask_nw in subtasks:
            subtasks[substask_nw].append(e)
        else:
            subtasks[substask_nw] = [e]

    print("extracted subtasks")
    final_subtasks = {}
    for s, e in subtasks.items():
        num_e = len(e)
        if num_e >= threshold_subtask:
            print(f"{s}: {len(e)}")
            final_subtasks[s] = e

    # for each subtask, compute top neurons and save
    for subtask, examples in final_subtasks.items():
        top_neurons = get_important_neurons(
            nnsight_model,
            examples,
            batch_size,
            mlps,
            pad_len,
            mean_act_files,
            task=task,
            noimg=noimg,
            flamingo=flamingo,
        )
        print(f"finished subtask: {subtask}")

        out_dir = f"results/top_neurons/{task}/{prefix}/"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(
            f"results/top_neurons/{task}/{prefix}/{subtask}_top_neurons.pkl", "wb"
        ) as f:
            pickle.dump(top_neurons, f)
