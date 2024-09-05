import torch
import gc
import sys
import os
import numpy as np
from tqdm import tqdm
import pickle
import json

from loading_utils import load_vqa_examples, load_blimp_examples, load_winoground_examples
from babylm_analysis import load_model, extract_submodules


def load_subtask(task_file, top_k):
    with open(f"{task_file}", "rb") as f:
        layer_dict = pickle.load(f)
        top_neurons = []
        for layer, tensors in layer_dict.items():
            top, effects = tensors
            top_new = [f"{layer}_{neuron}" for neuron in top]
            neuron_with_effect = [(neuron, effect) for neuron, effect in zip(top_new, effects)]
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
        pad_len,
        steps=10,
        metric_kwargs=dict()):
    tracer_kwargs = {'validate' : False, 'scan' : False}
    
    # clean run -> model can be approximated through linear function of its activations
    if img_inputs is None:
        with model.trace(clean, **tracer_kwargs), torch.no_grad():
            metric_clean = metric_fn(model, **metric_kwargs).save()

        with model.trace(clean, **tracer_kwargs), torch.no_grad():
            for layer_id, neurons in top_neurons_mean.items():
                for neuron_id in neurons:
                    model.git.encoder.layer[layer_id].intermediate[0].output[:,:,neuron_id] = top_neurons_mean[layer_id][neuron_id]
            metric_ablated = metric_fn(model, **metric_kwargs).save()
        
    else:
        with model.trace(clean, pixel_values=img_inputs, **tracer_kwargs), torch.no_grad(): 
            metric_clean = metric_fn(model, **metric_kwargs).save()

        with model.trace(clean, pixel_values=img_inputs, **tracer_kwargs), torch.no_grad():
            for layer_id, neurons in top_neurons_mean.items():
                for neuron_id in neurons:
                    model.git.encoder.layer[layer_id].intermediate[0].output[:,:,neuron_id] = top_neurons_mean[layer_id][neuron_id]
            metric_ablated = metric_fn(model, **metric_kwargs).save()
        
    total_effect = (metric_ablated.value - metric_clean.value).detach()

    return total_effect

def compute_metric_with_ablation(examples, batch_size, top_neurons_mean, task, noimg):
    num_examples = len(examples)
    batches = [examples[i:min(i + batch_size, num_examples)] for i in range(0, num_examples, batch_size)]
    device = "cuda"

    sum_effects = None

    for batch in tqdm(batches):
        clean_inputs = torch.cat([e['clean_prefix'] for e in batch], dim=0).to(device)

        if task == "vqa":
            clean_answer_idxs = torch.tensor([e['clean_answer'] for e in batch], dtype=torch.long, device=device)
            if noimg:
                img_inputs = None
            else:
                img_inputs = torch.cat([e['pixel_values'] for e in batch], dim=0).to(device)

            first_distractor_idxs = torch.tensor([e['distractors'][0] for e in batch], dtype=torch.long, device=device)

            def metric(model):
                # compute difference between correct answer and first distractor
                # TODO: compute avg difference between correct answer and each distractor
                #embds_out = model.output.output.save()
                
                return (
                    torch.gather(model.output.output[:,-1,:], dim=-1, index=first_distractor_idxs.view(-1, 1)).squeeze(-1) - \
                    torch.gather(model.output.output[:,-1,:], dim=-1, index=clean_answer_idxs.view(-1, 1)).squeeze(-1)
                )
            
            total_effect = patch_top_neurons(
                clean_inputs,
                img_inputs,
                model,
                top_neurons_mean,
                metric,
                pad_len,
                steps=10,
                metric_kwargs=dict())
        
        elif task == "blimp":
            img_inputs = None
            clean_answer_idxs = torch.tensor([e['clean_answer'] for e in batch], dtype=torch.long, device=device)
            patch_answer_idxs = torch.tensor([e['patch_answer'] for e in batch], dtype=torch.long, device=device)

            def metric(model):
                
                return (
                    torch.gather(model.output.output[:,-1,:], dim=-1, index=patch_answer_idxs.view(-1, 1)).squeeze(-1) - \
                    torch.gather(model.output.output[:,-1,:], dim=-1, index=clean_answer_idxs.view(-1, 1)).squeeze(-1)
                )
        

            total_effect = patch_top_neurons(
                clean_inputs,
                img_inputs,
                model,
                top_neurons_mean,
                metric,
                pad_len,
                steps=10,
                metric_kwargs=dict())
        
        elif task == "winoground":
            if noimg:
                img_inputs = None
            else:
                img_inputs = torch.cat([e['pixel_values'] for e in batch], dim=0).to(device)

            correct_idxs = [e["correct_idx"] for e in batch]
            incorrect_idxs = [e["incorrect_idx"] for e in batch]

            def metric(model):
                correct_sent_logits = []
                incorrect_sent_logits = []
                for i, (idx, cf_idx) in enumerate(zip(correct_idxs, incorrect_idxs)):
                    logits = torch.gather(model.output.output[i,:,:], dim=1, index=torch.tensor([idx]).to("cuda")).squeeze(-1) # [1, seq]
                    cf_logits = torch.gather(model.output.output[i,:,:], dim=1, index=torch.tensor([cf_idx]).to("cuda")).squeeze(-1) # [1, seq]
                    correct_sent_logits.append(logits.sum().unsqueeze(0))
                    incorrect_sent_logits.append(cf_logits.sum().unsqueeze(0))
                correct_sent_logits = torch.cat(correct_sent_logits, dim=0)
                incorrect_sent_logits = torch.cat(incorrect_sent_logits, dim=0)
                return incorrect_sent_logits-correct_sent_logits
            
            total_effect = patch_top_neurons(
                clean_inputs,
                img_inputs,
                model,
                top_neurons_mean,
                metric,
                pad_len,
                steps=10,
                metric_kwargs=dict())
                    
        if sum_effects is None:
            sum_effects = total_effect.sum(dim=0)
        else:
            sum_effects += total_effect.sum(dim=0)

        ablation_effect = sum_effects / num_examples
    return ablation_effect
    
if __name__ == "__main__":

    ############ Parse Config ##############
    with open(f"configs/{sys.argv[1]}", "r") as f:
        config = json.load(f)
    task = config["task"]
    num_examples = config["num_examples"]
    local = config["local"]
    print(f"task: {task} num_examples: {num_examples},  local: {local}")

    model_path = config["model_path"]
    epoch = config["epoch"]

    batch_size = config["batch_size"]
    pad_len = config["pad_len"]

    noimg = config["noimg"]
    threshold_subtask = config["threshold_subtask"]

    # load and prepare model
    model, tokenizer, img_processor = load_model(model_path, epoch, own_model=True, local=local)
    submodules = extract_submodules(model)
    mlps = [submodules[submodule] for submodule in submodules if submodule.startswith("mlp")]
    print("loaded model and submodules")

    # load and prepare data
    if task == "vqa":
        examples = load_vqa_examples(tokenizer, img_processor, pad_to_length=pad_len, n_samples=num_examples, local=local)
        subtask_key = "question_type"
    elif task == "blimp":
        examples = load_blimp_examples(tokenizer, pad_to_length=pad_len, n_samples=num_examples, local=local)
        subtask_key = "linguistics_term"
    elif task == "winoground":
        examples = load_winoground_examples(tokenizer, img_processor, pad_to_length=pad_len, n_samples=num_examples, local=local)
        subtask_key = "secondary_tag"
    else:
        print(f"{task} is not implemented")
    print("loaded samples")

    # retrieve precomputed mean activation files
    mean_act_files=[]
    prefix = f"{task}_{model_path}_e{epoch}_n{num_examples if num_examples != -1 else 'all'}{'_noimg' if noimg else ''}"
    for file in os.listdir("mean_activations/"):
        if file.startswith(prefix+"_mean_acts"):
            mean_act_files.append(f"mean_activations/{file}")
    if local == False and len(mean_act_files) != len(mlps):
        raise Exception
    print("loaded mean activations")
    
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
    
    # iterate over subtasks and ablate their top neurons
    for subtask, examples in final_subtasks.items():
        print(f"ablating top neurons of subtask: {subtask}")
        all_ablation_effects = {}
        # get top neurons
        top_neurons_file = f"data/top_neurons/{task}/{prefix}/{subtask}_top_neurons.pkl"
        top_neurons = load_subtask(top_neurons_file, top_k=100)
        # get their top activations
        top_neurons_mean = {}
        for layer, neurons in top_neurons.items():
            if local and layer > len(mean_act_files)-1:
                print(f"skipping ablation in layer: {layer}")
                continue
            top_neurons_mean[layer] = {}
            mean_state = np.load(mean_act_files[layer], allow_pickle=True)
            mean_state = torch.tensor(mean_state).to("cuda")
            for neuron in neurons:
                # TBD: eventuell hat mean state nicht nur n_dim dimensionen
                top_neurons_mean[layer][neuron] = mean_state[neuron]
            mean_state = None
            del mean_state
            torch.cuda.empty_cache()
            gc.collect()
        
        # do attribution patching with selected top neurons per subtask
        for eval_subtask, eval_examples in final_subtasks.items():
            print(f"evaluating subtask: {eval_subtask}")
            ablation_effect = compute_metric_with_ablation(eval_examples, batch_size, top_neurons_mean, task, noimg)
            all_ablation_effects[eval_subtask] = ablation_effect
        
        top_neurons_mean = None
        del top_neurons_mean
        torch.cuda.empty_cache()
        gc.collect()

        out_dir = f"data/ablation_top_neurons/{task}/{prefix}/"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(f"data/ablation_top_neurons/{task}/{prefix}/{subtask}_top_neurons_ablated.pkl", "wb") as f:
            pickle.dump(all_ablation_effects, f)

