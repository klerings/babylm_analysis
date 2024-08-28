from nnsight import LanguageModel

import torch as t
import gc
import sys
import math
import numpy as np
from tqdm import tqdm
import torch
from datasets import load_dataset

from loading_utils import load_vqa_examples, load_blimp_examples

from transformers import AutoProcessor, AutoTokenizer
from nnsight import NNsight
import importlib.util
import pickle
import sys


def load_model(model_path, epoch, own_model=True, local=False):
    if own_model:
        model_path = f"../babylm_GIT/{'models_for_eval' if not local else 'models2'}/base_{model_path}/epoch{epoch}/"
        spec = importlib.util.spec_from_file_location("GitForCausalLM", f"{model_path}modeling_git.py")
        git_module = importlib.util.module_from_spec(spec)
        sys.modules["git_module"] = git_module
        spec.loader.exec_module(git_module)
        GitForCausalLM = git_module.GitForCausalLM

        model = GitForCausalLM.from_pretrained(model_path) 
        ckpt = torch.load(model_path + "pytorch_model.bin") # TODO: newly initialized for vision encoder: ['pooler.dense.bias', 'pooler.dense.weight']
        model.load_state_dict(ckpt, strict=False)  
        
    else:
        model_path = "babylm/git-2024"

        from transformers import GitForCausalLM as OGModel

        model = OGModel.from_pretrained(model_path, trust_remote_code=True)
        
    # load tokenizer and img processor
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    img_processor = AutoProcessor.from_pretrained(model_path,trust_remote_code=True)
    
    nnsight_model = NNsight(model, device_map="cuda")
    nnsight_model.to("cuda")

    return nnsight_model, tokenizer, img_processor


def extract_submodules(model):
    submodules = {}
    for idx, layer in enumerate(model.git.encoder.layer):
        submodules[f"mlp.{idx}"] = layer.intermediate    # output of MLP
        submodules[f"attn.{idx}"] = layer.attention  # output of attention
        submodules[f"resid.{idx}"] = layer      # output of whole layer
    return submodules

def compute_mean_activations(examples, model, submodules, batch_size, num_examples, noimg=False):
    tracer_kwargs = {'validate' : False, 'scan' : False}
    device = "cuda"
    num_examples = min([num_examples, len(examples)])
    batches = [
        examples[i:min(i + batch_size,num_examples)] for i in range(0, num_examples, batch_size)
    ]

    # Initialize storage for cumulative activations and count of samples
    cumulative_activations = {submodule: 0 for submodule in submodules}
    total_samples = 0

    for batch in tqdm(batches):
        clean_inputs = t.cat([e['clean_prefix'] for e in batch], dim=0).to(device)
    
        # clean run -> model can be approximated through linear function of its activations
        hidden_states_clean = {}
        if noimg:
            with model.trace(clean_inputs, **tracer_kwargs), t.no_grad():
                for submodule in submodules:
                    x = submodule.output
                    hidden_states_clean[submodule] = x.save()
        else:
            img_inputs = t.cat([e['pixel_values'] for e in batch], dim=0).to(device)
            with model.trace(clean_inputs, pixel_values=img_inputs, **tracer_kwargs), t.no_grad():
                for submodule in submodules:
                    x = submodule.output
                    hidden_states_clean[submodule] = x.save()
        hidden_states_clean = {k : v.value for k, v in hidden_states_clean.items()}

        batch_size = next(iter(hidden_states_clean.values())).shape[0]  # Assuming shape [batch_size, ...]
        total_samples += batch_size

        # Sum across the batch (dim=0)
        for submodule, state in hidden_states_clean.items():
            cumulative_activations[submodule] += state.sum(dim=(0, 1))  
        
        hidden_states_clean = None
        torch.cuda.empty_cache()
        gc.collect()

    # Compute mean activation by dividing the cumulative activations by the total number of samples
    mean_activations = {submodule: cum_act / total_samples for submodule, cum_act in cumulative_activations.items()}

    return mean_activations

# Attribution patching with integrated gradients
def _pe_ig(
        clean,
        img_inputs,
        model,
        submodules,
        hidden_states_mean,
        metric_fn,
        steps=10,
        metric_kwargs=dict()):
    tracer_kwargs = {'validate' : False, 'scan' : False}
    
    # clean run -> model can be approximated through linear function of its activations
    hidden_states_clean = {}
    if img_inputs is None:
        with model.trace(clean, **tracer_kwargs), t.no_grad():
            for submodule in submodules:
                x = submodule.output
                hidden_states_clean[submodule] = x.save()
            metric_clean = metric_fn(model, **metric_kwargs).save()
    else:
        with model.trace(clean, pixel_values=img_inputs, **tracer_kwargs), t.no_grad(): 
            for submodule in submodules:
                x = submodule.output
                hidden_states_clean[submodule] = x.save()
            metric_clean = metric_fn(model, **metric_kwargs).save()
    hidden_states_clean = {k : v.value for k, v in hidden_states_clean.items()}


    effects = {}
    deltas = {}
    grads = {}
    for submodule in submodules:
        clean_state = hidden_states_clean[submodule]
        mean_state = hidden_states_mean[submodule]
        with model.trace(**tracer_kwargs) as tracer:
            metrics = []
            fs = []
            for step in range(steps):
                alpha = step / steps
                f = (1 - alpha) * clean_state + alpha * mean_state
                f.retain_grad()
                fs.append(f)
                if img_inputs is None:
                    with tracer.invoke(clean, scan=tracer_kwargs['scan']):
                        submodule.output = f
                        metrics.append(metric_fn(model, **metric_kwargs))
                else:
                    with tracer.invoke(clean, pixel_values=img_inputs, scan=tracer_kwargs['scan']):
                        submodule.output = f
                        metrics.append(metric_fn(model, **metric_kwargs))
            metric = sum([m for m in metrics])
            metric.sum().backward(retain_graph=True) # TODO : why is this necessary? Probably shouldn't be, contact jaden
        
        mean_grad = sum([f.grad for f in fs]) / steps
        # mean_residual_grad = sum([f.grad for f in fs]) / steps
        grad = mean_grad
        delta = (mean_state - clean_state).detach() if mean_state is not None else -clean_state.detach()
        effect = t.mul(grad, delta)

        effects[submodule] = effect
        deltas[submodule] = delta
        grads[submodule] = grad
    
    return (effects, deltas, grads)

def get_important_neurons(examples, batch_size, mlps, mean_activations, task):
    # uses attribution patching to identify most important neurons for subtask
    num_examples = len(examples)
    batches = [examples[i:min(i + batch_size, num_examples)] for i in range(0, num_examples, batch_size)]
    device = "cuda"

    sum_effects = {}

    for batch in tqdm(batches):
        clean_answer_idxs = t.tensor([e['clean_answer'] for e in batch], dtype=t.long, device=device)
        clean_inputs = t.cat([e['clean_prefix'] for e in batch], dim=0).to(device)

        if task == "vqa":
            img_inputs = t.cat([e['pixel_values'] for e in batch], dim=0).to(device)

            first_distractor_idxs = t.tensor([e['distractors'][0] for e in batch], dtype=t.long, device=device)

            def metric(model):
                # compute difference between correct answer and first distractor
                # TODO: compute avg difference between correct answer and each distractor
                #embds_out = model.output.output.save()
                
                return (
                    t.gather(model.output.output[:,-1,:], dim=-1, index=first_distractor_idxs.view(-1, 1)).squeeze(-1) - \
                    t.gather(model.output.output[:,-1,:], dim=-1, index=clean_answer_idxs.view(-1, 1)).squeeze(-1)
                )
        
        elif task == "blimp":
            img_inputs = None
            
            patch_answer_idxs = t.tensor([e['patch_answer'] for e in batch], dtype=t.long, device=device)

            def metric(model):
                
                return (
                    t.gather(model.output.output[:,-1,:], dim=-1, index=patch_answer_idxs.view(-1, 1)).squeeze(-1) - \
                    t.gather(model.output.output[:,-1,:], dim=-1, index=clean_answer_idxs.view(-1, 1)).squeeze(-1)
                )
        

        effects, _, _ = _pe_ig(
            clean_inputs,
            img_inputs,
            model,
            mlps,
            mean_activations,
            metric,
            steps=10,
            metric_kwargs=dict())
        
        
        for submodule in mlps:
            if submodule not in sum_effects:
                sum_effects[submodule] = effects[submodule].sum(dim=1).sum(dim=0)
            else:
                sum_effects[submodule] += effects[submodule].sum(dim=1).sum(dim=0)

    # Print top 1% neurons in each submodule (ndim=3072)
    k = 31

    top_neurons = {}
    for idx, submodule in enumerate(mlps):
        sum_effects[submodule] /= num_examples
        v, i = t.topk(sum_effects[submodule].flatten(), k)  # v=top effects, i=top indices
        top_neurons[f"mlp_{idx}"] = (i,v)
    return top_neurons
        


if __name__ == "__main__":
    task = sys.argv[1]

    model_path = "git_1vd125_s1"
    epoch = 23
    local = False
    # load and prepare model
    model, tokenizer, img_processor = load_model(model_path, epoch, own_model=True, local=local)
    submodules = extract_submodules(model)
    mlps = [submodules[submodule] for submodule in submodules if submodule.startswith("mlp")]

    print("loaded model and submodules")

    batch_size = 8
    num_examples = -1
    noimg = False

    # load and prepare data
    if task == "vqa":
        examples = load_vqa_examples(tokenizer, img_processor, pad_to_length=32, n_samples=num_examples, local=local)
        subtask_key = "question_type"
    elif task == "blimp":
        noimg = True
        examples = load_blimp_examples(tokenizer, pad_to_length=32, n_samples=num_examples)
        subtask_key = "UID"
    else:
        print(f"{task} is not implemented")
    print("loaded samples")

    # precompute mean activations on task
    mean_activations = compute_mean_activations(examples, model, mlps, batch_size, num_examples, noimg=noimg)

    print(f"computed mean activations")

    # identify subtasks
    subtasks = {}
    for e in examples:
        subtask = e[subtask_key]
        if subtask in subtasks:
            subtasks[subtask].append(e)
        else:
            subtasks[subtask] = [e]

    print("extracted subtasks")

    # for each subtask, compute top neurons and save
    subtasks_neurons = {}
    for subtask, examples in subtasks.items():
        top_neurons = get_important_neurons(examples, batch_size, mlps, mean_activations, task=task)
        subtasks_neurons[subtask] = top_neurons
        print(f"finished subtask: {subtask}")

    with open(f"data/{model_path}_e{epoch}_{task}_top_neurons_per_subtask.pkl", "wb") as f:
        pickle.dump(subtasks_neurons, f)