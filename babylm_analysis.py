from nnsight import LanguageModel

import torch as t
import gc
import sys
import math
import numpy as np
import os
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast
from datasets import load_dataset

from loading_utils import load_vqa_examples, load_blimp_examples

from transformers import AutoProcessor, AutoTokenizer
from nnsight import NNsight
import importlib.util
import pickle
import sys
import json

from memory_profiler import profile


def load_model(model_path, epoch, own_model=True, local=False):
    if own_model:
        if local:
            model_path = f"../babylm_GIT/models2/base_{model_path}/epoch{epoch}/"
        else:
            model_path = f"/home/ma/ma_ma/ma_aklering/gpfs/ma_aklering-babylm2/babylm_GIT/models_for_eval/base_{model_path}/epoch{epoch}/"
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

def compute_mean_activations(examples, model, submodules, batch_size, noimg=False, file_prefix=None):
    tracer_kwargs = {'validate' : False, 'scan' : False}
    device = "cuda"
    num_examples = len(examples)
    batches = [
        examples[i:min(i + batch_size,num_examples)] for i in range(0, num_examples, batch_size)
    ]

    def extract_hidden_states(submodule):
        total_samples = 0
        # Initialize storage for cumulative activations and count of samples
        cumulative_activations = 0

        for batch in tqdm(batches):
            clean_inputs = t.cat([e['clean_prefix'] for e in batch], dim=0).to(device)
        
            # clean run -> model can be approximated through linear function of its activations
            hidden_states_clean = {}
            #with autocast():
            if noimg:
                with model.trace(clean_inputs, **tracer_kwargs), t.no_grad():
                    x = submodule.output
                    hidden_states_clean = x.save()
            else:
                img_inputs = t.cat([e['pixel_values'] for e in batch], dim=0).to(device)
                with model.trace(clean_inputs, pixel_values=img_inputs, **tracer_kwargs), t.no_grad():
                    x = submodule.output
                    hidden_states_clean = x.save()
            hidden_states_clean = hidden_states_clean.value

            batch_size = clean_inputs.shape[0]  # Assuming shape [batch_size, ...]
            total_samples += batch_size

            # Sum across the batch (dim=0)
            cumulative_activations += hidden_states_clean.sum(dim=(0, 1)).detach().cpu()  # detach
            
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
#@profile
def _pe_ig(
        clean,
        img_inputs,
        model,
        submodules,
        mean_act_files,
        metric_fn,
        pad_len,
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
                    with tracer.invoke(clean, scan=tracer_kwargs['scan']):
                        submodule.output = f
                        metrics.append(metric_fn(model, **metric_kwargs))
                else:
                    with tracer.invoke(clean, pixel_values=img_inputs, scan=tracer_kwargs['scan']):
                        submodule.output = f[:,-pad_len:,:]   # we are only interested in the text tokens which are appended to img tokens
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

        mean_state = None
        del mean_state
        torch.cuda.empty_cache()
        gc.collect()

    
    return (effects, deltas, grads)

def _pe_ig_patch(
        clean,
        patch,
        model,
        submodules,
        metric_fn,
        pad_len,
        steps=10,
        metric_kwargs=dict()):
    tracer_kwargs = {'validate' : False, 'scan' : False}
    
    # clean run -> model can be approximated through linear function of its activations
    hidden_states_clean = {}
    with model.trace(clean, **tracer_kwargs), t.no_grad():
        for submodule in submodules:
            x = submodule.output
            hidden_states_clean[submodule] = x.save()
        metric_clean = metric_fn(model, **metric_kwargs).save()

    hidden_states_clean = {k : v.value for k, v in hidden_states_clean.items()}

    x = None
    del x
    torch.cuda.empty_cache()
    gc.collect()

    # patch run
    hidden_states_patch = {}
    with model.trace(patch, **tracer_kwargs), t.no_grad():
        for submodule in submodules:
            x = submodule.output
            hidden_states_patch[submodule] = x.save()
        metric_patch = metric_fn(model, **metric_kwargs).save()

    hidden_states_patch = {k : v.value for k, v in hidden_states_patch.items()}

    x = None
    del x
    torch.cuda.empty_cache()
    gc.collect()

    effects = {}
    deltas = {}
    grads = {}
    for i, submodule in enumerate(submodules):        
        patch_state = hidden_states_patch[submodule]
        clean_state = hidden_states_clean[submodule]
        with model.trace(**tracer_kwargs) as tracer:
            metrics = []
            fs = []
            for step in range(steps):
                alpha = step / steps
                f = (1 - alpha) * clean_state + alpha * patch_state
                f.retain_grad()
                fs.append(f)
                with tracer.invoke(clean, scan=tracer_kwargs['scan']):
                    submodule.output = f[:,-pad_len:,:]
                    metrics.append(metric_fn(model, **metric_kwargs))
                
            metric = sum([m for m in metrics])
            metric.sum().backward(retain_graph=True) # TODO : why is this necessary? Probably shouldn't be, contact jaden
        
        mean_grad = sum([f.grad for f in fs]) / steps
        # mean_residual_grad = sum([f.grad for f in fs]) / steps
        grad = mean_grad
        delta = (patch_state - clean_state).detach() if patch_state is not None else -clean_state.detach()
        effect = t.mul(grad, delta)

        effects[submodule] = effect
        deltas[submodule] = delta
        grads[submodule] = grad

        patch_state = None
        del patch_state
        torch.cuda.empty_cache()
        gc.collect()

    
    return (effects, deltas, grads)

def get_important_neurons(examples, batch_size, mlps, pad_len, mean_act_files, task):
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
            
            effects, _, _ = _pe_ig(
                clean_inputs,
                img_inputs,
                model,
                mlps,
                mean_act_files,
                metric,
                pad_len,
                steps=10,
                metric_kwargs=dict())
        
        elif task == "blimp":
            img_inputs = None
            patch_inputs = t.cat([e['patch_prefix'] for e in batch], dim=0).to(device)
            
            patch_answer_idxs = t.tensor([e['patch_answer'] for e in batch], dtype=t.long, device=device)

            def metric(model):
                
                return (
                    t.gather(model.output.output[:,-1,:], dim=-1, index=patch_answer_idxs.view(-1, 1)).squeeze(-1) - \
                    t.gather(model.output.output[:,-1,:], dim=-1, index=clean_answer_idxs.view(-1, 1)).squeeze(-1)
                )
        

            effects, _, _ = _pe_ig_patch(
                clean_inputs,
                patch_inputs,
                model,
                mlps,
                metric,
                pad_len,
                steps=10,
                metric_kwargs=dict())
        
        
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
        v, i = t.topk(sum_effects[submodule].flatten(), k)  # v=top effects, i=top indices
        top_neurons[f"mlp_{idx}"] = (i.cpu(),v.cpu())
    return top_neurons
        


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
        mean_act_files=None
    else:
        print(f"{task} is not implemented")
    print("loaded samples")

    # precompute mean activations on task or retrieve precomputed activation files
    prefix = f"{task}_{model_path}_e{epoch}_n{num_examples if num_examples != -1 else 'all'}{'_noimg' if noimg else ''}"
    if task != "blimp":
        mean_act_files = []
        for file in os.listdir("mean_activations/"):
            if file.startswith(prefix+"_mean_acts"):
                mean_act_files.append(f"mean_activations/{file}")
        if len(mean_act_files) == 0:
            mean_act_files = compute_mean_activations(examples, model, mlps, batch_size=128, noimg=noimg, file_prefix=prefix)
            print(f"computed mean activations")
        else:
            print("retrieved precomputed mean activations")
    else:
        print("using counterfactuals instead of mean activations")

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
        top_neurons = get_important_neurons(examples, batch_size, mlps, pad_len, mean_act_files, task=task)
        print(f"finished subtask: {subtask}")

        out_dir = f"data/top_neurons/{task}/{prefix}/"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(f"data/top_neurons/{task}/{prefix}/{subtask}_top_neurons.pkl", "wb") as f:
            pickle.dump(top_neurons, f)