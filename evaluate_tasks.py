import torch
from datasets import load_dataset, Dataset, concatenate_datasets
import json
from tqdm import tqdm
import pickle
import sys
import os
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoProcessor
import torch.nn.functional as F
import numpy as np
from PIL import Image
import gc
import re
import importlib.util


def load_blimp(task, n_samples=32, load_all=False):
    final_samples = {}
    file_dir = f"../evaluation-pipeline-2024/evaluation_data/{task}/"
    subtasks = [file_dir + f for f in os.listdir(file_dir)]
    datasets = []
    n_subtasks = 67 if task == "blimp_filtered" else 5
    n_samples_subtask = max(int(round(n_samples/n_subtasks)),1)
    for subtask in subtasks:
        ds = load_dataset(path="json", name=None, data_files=subtask)["train"]
        shuffled_ds = ds.shuffle(seed=42)
        if load_all:
            datasets.append(shuffled_ds)
        elif n_samples_subtask < len(shuffled_ds):
            datasets.append(shuffled_ds.select(range(n_samples_subtask)))
        else:
            datasets.append(shuffled_ds)
    full_dataset = concatenate_datasets(datasets)
    for i, sample in enumerate(full_dataset):
        final_samples[i] = sample
        final_samples[i]["prompts"] = [("", f" {sample['sentence_good']}"), ("", f" {sample['sentence_bad']}")]
    return final_samples

def split_options2(question):
    q = question.replace("Question:", "")
    q = re.sub(r"Hint:.+?\n", "", q)
    parts = re.split(r"\(A\)|\(B\)|\(C\)|\(D\)", q)
    quest = parts[0].strip().replace("Choices:", "").replace("Options:", "").strip()
    opts = [o.strip().rstrip(",.") for o in parts[1:]]
    assert len(opts) >= 3 and len(opts) < 5, question
    return (quest.strip(), opts)

def split_options(question):
    if "Choices:" in question:
        #print(split_options2(question))
        #exit()
        return split_options2(question)
    q = question.replace("Question:", "")
    q = re.sub(r"Hint:.+?\n", "", question)
    parts = re.split(r"A: |B: |C: |D: ", q)
    quest = parts[0].strip().replace("Options:", "").strip()
    opts = [o.strip().rstrip(",.") for o in parts[1:]]
    if not len(opts) >= 3 or not len(opts) < 5:
        return split_options2(question)
    return (quest.strip(), opts)

def load_mmstar(n_samples):
    data = load_dataset("Lin-Chen/MMStar")["val"]
    if n_samples < len(data):
        data = data.select(range(n_samples))
    final_samples = {}
    for i, sample in enumerate(tqdm(data)):
        question, options = split_options(sample['question'])
        answer_idx = ord(sample["answer"])-ord("A")
        sample['question'] = question
        ordered_options = [options.pop(answer_idx)] + options[:4]

        final_samples[i] = sample
        final_samples[i]["prompts"] = []
        for oo in ordered_options:
            final_samples[i]["prompts"].append(transform_qa(question, oo))
    
    return final_samples
        


def load_vl_data(task="vqa", n_samples=32, local=False):
    if task == "vqa":
        hf_path = "HuggingFaceM4/VQAv2"
        hf_split = "validation"
        local_file = f"data/vqa_filtered/vqa_distractors_info.json"
    elif task == "winoground":
        hf_path = "facebook/winoground"
        hf_split = "test"
        local_file = f"data/winoground_filtered/winoground.jsonl"
    
    # load huggingface dataset with images
    if not local:
        hf_ds = load_dataset(hf_path, cache_dir="/home/ma/ma_ma/ma_aklering/gpfs/ma_aklering-babylm2/.cache/")[hf_split]
    else:
        hf_ds = load_dataset(hf_path)[hf_split]

    print("loaded huggingface DS")
        
    # load dataset with distractors from json
    distractor_ds = load_dataset(path="json", name=None, data_files=local_file)["train"]

    print("loaded local DS")
    shuffled_distractors = distractor_ds.shuffle(seed=42)
    if n_samples < len(shuffled_distractors) and n_samples > 0:
        shuffled_distractors = shuffled_distractors.select(range(n_samples))

    # merge datasets by adding distractor info from json to hf dataset samples
    samples = []
    for sample in tqdm(shuffled_distractors):
        if task == "vqa":
            # add distractor info from json to hf dataset sample
            img_sample = hf_ds[sample["idx_in_hf_dataset"]] 
            if img_sample["image_id"] != sample["image_id"]:
                print(f"Sample Mismatch: {sample['image_id']} - {img_sample['image_id']}")
            else:
                img_sample["distractors"] = sample["distractors"]
                samples.append(img_sample)
        elif task == "winoground":
            # add image to json dataset samples because winoground batches pairs together as one sample and json file splits them
            img_sample = hf_ds[sample["image_idx"]]
            img = img_sample[sample["image_key"]]
            sample["image"] = img
            samples.append(sample)

    final_samples = {}
    # per sample create prompts consisting of tuple with question-answer-prompt (context) and answer-options (continuation)
    for sample in samples:
        if task == "vqa":
            final_samples[sample["question_id"]] = sample
            final_samples[sample["question_id"]]["prompts"]= []
            final_samples[sample["question_id"]]["prompts"].append(transform_qa(sample["question"], sample["multiple_choice_answer"]))
            for d in sample["distractors"]:
                final_samples[sample["question_id"]]["prompts"].append(transform_qa(sample["question"], d))
        elif task == "winoground":
            final_samples[sample["id"]] = sample
            final_samples[sample["id"]]["prompts"] = [("", f" {sample['caption_0']}"), ("", f" {sample['caption_1']}")]
   
    return final_samples


# prepare input format (img + question + answer)
def transform_qa(question, answer):
    return f"Question: {question} Answer:", f" {answer}"

def load_model(setting, epoch, local=False):
    # Add the directory containing the modules to the Python path
    model_path = f"../babylm_GIT/{'models_for_eval' if not local else 'models2'}/base_{setting}/epoch{epoch}/"
    if not os.path.exists(model_path+"pytorch_model.bin"):
        return None, None, None
    spec = importlib.util.spec_from_file_location("GitForCausalLM", f"{model_path}modeling_git.py")
    git_module = importlib.util.module_from_spec(spec)
    sys.modules["git_module"] = git_module
    spec.loader.exec_module(git_module)
    GitForCausalLM = git_module.GitForCausalLM

    model = GitForCausalLM.from_pretrained(model_path) 
    ckpt = torch.load(model_path + "pytorch_model.bin") # TODO: newly initialized for vision encoder: ['pooler.dense.bias', 'pooler.dense.weight']
    model.load_state_dict(ckpt, strict=False)  

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if setting.startswith("git"):
        print("loading processor")
        image_processor = AutoProcessor.from_pretrained(
                            model_path,
                            trust_remote_code=True
                        )
    else:
        print("not loading processor")
        image_processor = None

    device = torch.device("cuda")
    model.to(device)
    return model, image_processor, tokenizer
 
def pad_and_concat(
    max_length: int,
    tensors,
    padding_side = "right",
):
    """
    Method for padding a list of tensors given the maximum tensor
    length in the batch. Used for batching inputs and continuations in
    seq2seq models.
    """
    assert (
        padding_side == "left" or padding_side == "right"
    ), f"Unrecognized padding type: '{padding_side}' not 'left' or 'right'"

    for i, tensor in enumerate(tensors):
        if len(tensor.shape) == 2:
            tensor = tensor.squeeze(0)  # squeeze, in case passed [1, seq] size
        tensor_len = tensor.shape[0]
        if tensor_len < max_length:
            if padding_side == "right":
                # right-pad
                tensors[i] = torch.cat(
                    [
                        tensor,  # [seq]
                        torch.zeros(
                            max_length - tensor_len,
                            dtype=torch.long,
                            device=tensor.device,
                        ),  # [padding_length - seq]
                    ],
                    dim=0,
                ).unsqueeze(0)
            else:
                # left-pad
                tensors[i] = torch.cat(
                    [
                        torch.zeros(
                            max_length - tensor_len,
                            dtype=torch.long,
                            device=tensor.device,
                        ),  # [padding_length - seq]
                        tensor,  # [seq]
                    ],
                    dim=0,
                ).unsqueeze(0)
        else:
            tensors[i] = tensor.unsqueeze(0)

    return torch.cat(tensors, dim=0)

def eot_token_id(tokenizer):
    # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = tokenizer.pad_token_id
        return tokenizer, tokenizer.pad_token_id
    return tokenizer, tokenizer.eos_token_id

def prepare_sample(prompt, image, tokenizer, image_processor, padding_len_inp, mode, task):
    max_length = 1024
    context, continuation = prompt
    if task in ["vqa","mmstar"]:
        whole_enc = tokenizer.encode(context + continuation, add_special_tokens=False)
        context_enc = tokenizer.encode(context, add_special_tokens=False)
        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]
    elif task in ["winoground", "blimp_filtered", "supplement_filtered"]:
        tokenizer, prefix_token_id = eot_token_id(tokenizer)
        context_enc = [prefix_token_id]
        continuation_enc = tokenizer.encode(continuation, add_special_tokens=False)
    else:
        raise Exception(f"{task} not implemented")
    
    inp = torch.tensor(
                            (context_enc + continuation_enc)[-(max_length + 1) :][:-1],
                            dtype=torch.long,
                            device=torch.device("cuda"),
                )
    (inplen,) = inp.shape
    padding_len_inp = (
                        max(padding_len_inp, inplen)
                        if padding_len_inp is not None
                        else inplen
                    )
    
    if mode == "with_img":
        image_embs = image_processor(images=image.convert(mode="RGB"), return_tensors="pt")["pixel_values"].to(torch.device("cuda"))
    elif mode == "with_noise":
        width, height = image.size
        noise_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        noise_img = Image.fromarray(noise_array, mode='RGB')
        image_embs = image_processor(images=noise_img.convert(mode="RGB"), return_tensors="pt")["pixel_values"].to(torch.device("cuda"))
    else:
        image_embs = None

    return inp, image_embs, inplen, continuation_enc, prompt, padding_len_inp

def run_eval(samples, model, image_processor, tokenizer, mode, task, batch_size=32):
    
    padding_len_inp = None
    total_acc = 0
    batch_size = 16

    batched_images = []
    batched_inps = []
    batched_inplens = []
    batched_cont_toks_list = []
    batched_prompt_list = []
    all_results = []

    num_prompts = None
    for question_id, question_dict in tqdm(samples.items()):
        images = []
        inps = []
        inplens = []
        cont_toks_list = []

        image = question_dict["image"] if "image" in question_dict else None
        prompts = question_dict["prompts"]
        num_prompts = len(prompts)

        for prompt in prompts:
            inp, image_embs, inplen, continuation_enc, prompt, padding_len_inp = prepare_sample(prompt, image, tokenizer, image_processor, padding_len_inp, mode, task)
            
            images.append(image_embs)
            inps.append(inp)
            inplens.append(inplen)
            cont_toks_list.append(continuation_enc)
            batched_prompt_list.append(prompt)

        batched_images.extend(images)
        batched_inps.extend(inps)
        batched_inplens.extend(inplens)
        batched_cont_toks_list.extend(cont_toks_list)
        
        # If the accumulated batch size reaches the desired size, or if it's the last sample:
        if len(batched_inps) >= batch_size or question_id == list(samples.keys())[-1]:
            # Pad and concatenate inputs and images
            batched_inps = pad_and_concat(padding_len_inp, batched_inps, padding_side="right")

            if mode == "with_img" or mode == "with_noise":
                batched_images = torch.cat(batched_images, dim=0)
            
                # Run the model on the batched inputs and images
                multi_logits = F.log_softmax(
                    model(batched_inps, pixel_values=batched_images).logits, dim=-1
                )  # [batch, padding_length (inp or cont), vocab]
            
            else:
                 multi_logits = F.log_softmax(
                    model(batched_inps).logits, dim=-1
                )  # [batch, padding_length (inp or cont), vocab]

            # Calculate log-likelihoods for each prompt in the batch
            batch_results = []
            for logits, inplen, cont_toks in zip(multi_logits, batched_inplens, batched_cont_toks_list):
                # slice to original seq length
                contlen = len(cont_toks)
                ctx_len = (inplen + (logits.shape[0] - padding_len_inp))
                logits = logits[ctx_len - contlen : ctx_len]
                logits = logits.unsqueeze(0)  # [1, seq, vocab]
                
                cont_toks = torch.tensor(
                                cont_toks, dtype=torch.long, device=torch.device("cuda")
                            ).unsqueeze(0)  # [1, seq]
                logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(
                    -1
                )  # [1, seq]
                answer = float(logits.sum())
                batch_results.append(answer)

            all_results.extend(batch_results)
            
            # Reset the batched variables for the next set of samples
            batched_images = []
            batched_inps = []
            batched_inplens = []
            batched_cont_toks_list = []
            batched_prompt_list = []

    # Compare results with gold standard and compute accuracy
    for i in range(0, len(all_results), num_prompts):
        gold = 0  # first prompt is always the gold prompt
        acc = 1.0 if np.argmax(all_results[i:i+num_prompts]) == gold else 0.0
        total_acc += acc

    final_acc = total_acc / len(samples)
    return final_acc



if __name__ == "__main__":
    subset_size = -1
    batch_size = 64
    local=False

    txt_only_tasks = ["blimp_filtered", "supplement_filtered"]
    vl_tasks = ["mmstar", "vqa", "winoground"]
    tasks = ["vqa"]    #vl_tasks + txt_only_tasks

    settings = ["git_1v1_s1", "git_1vd5_s1", "git_1vd25_s1", "git_1vd125_s1", "git_1v1_s2", "git_1vd5_s2", "git_1vd25_s2", "git_1vd125_s2", "git_1v1_s3", "git_1vd5_s3", "git_1vd25_s3", "git_1vd125_s3"]

    for setting in settings:
        for epoch in range(1,31):

            model, image_processor, tokenizer = load_model(setting, epoch, local)
            if model is None:
                print(f"skipping: {setting} - {epoch}")
                continue

            for task in tasks:

                if task in txt_only_tasks:
                    samples = load_blimp(task, n_samples=subset_size)
                elif task == "mmstar":
                    samples = load_mmstar(n_samples=subset_size)
                else:
                    samples = load_vl_data(task, n_samples=subset_size, local=local)
                
                acc_no_img = run_eval(samples, model, image_processor, tokenizer, mode="txt_only", task=task, batch_size=batch_size)
                print(f"\n{setting} - {task}")
                print(f"-> no img: {acc_no_img}")

                torch.cuda.empty_cache()
                gc.collect()

                if setting.startswith("git") and task not in txt_only_tasks:

                    acc_with_img = run_eval(samples, model, image_processor, tokenizer, mode="with_img",  task=task, batch_size=batch_size)
                    print(f"\n{setting} - {task}")
                    print(f"-> with img: {acc_with_img}")
                    
                    torch.cuda.empty_cache()
                    gc.collect()

                    #acc_with_noise = run_eval(samples, model, image_processor, tokenizer, mode="with_noise",  task=task, batch_size=batch_size)
                    #print(f"-> with noise: {acc_with_noise}")

                    #torch.cuda.empty_cache()
                    #gc.collect()
        



    
