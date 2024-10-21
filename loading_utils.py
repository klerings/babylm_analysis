import random
import os
import re
import sys
from tqdm import tqdm
import torch
import torch.nn.functional as F
from datasets import load_dataset, concatenate_datasets
from string import punctuation
from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM
from nnsight import NNsight
import importlib.util

def load_blimp(task, n_samples=32, load_all=False):
    final_samples = {}
    file_dir = f"data/{task}/"
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
    if n_samples < len(data)  and n_samples > 0:
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
        
def load_ewok(n_samples, load_all=False):
    final_samples = {}
    file_dir = "data/ewok_filtered/"
    subtasks = [file_dir + f for f in os.listdir(file_dir)]
    datasets = []
    n_subtasks = 11
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
    return final_samples

def load_vl_data(task="vqa", n_samples=32):
    if task == "vqa":
        hf_path = "HuggingFaceM4/VQAv2"
        hf_split = "validation"
        local_file = f"data/vqa_filtered/vqa_distractors_info.json"
    elif task == "winoground":
        hf_path = "facebook/winoground"
        hf_split = "test"
        local_file = f"data/winoground_filtered/winoground.jsonl"
    
    # load huggingface dataset with images
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

def load_vqa_examples(tokenizer, img_processor, pad_to_length, n_samples):
    samples = load_vl_data(task="vqa", n_samples=n_samples*10)
    examples = []
    eos_token_id = tokenizer.convert_tokens_to_ids("</s>")
    lookup = parse_vqa_qtypes()
    for sample_id, sample in samples.items():
        clean_tokens = tokenizer(sample["question"], return_tensors="pt",
                                        padding=False)
        clean_prefix = clean_tokens.input_ids
        # add answer with preceding whitespace
        clean_answer = tokenizer(f" {sample['multiple_choice_answer']}", return_tensors="pt",
                                        padding=False).input_ids
        distractors = [tokenizer(f" {d}", return_tensors="pt",
                                        padding=False).input_ids for d in sample["distractors"]]
        
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
        if pad_to_length:
            tokenizer.padding_side = 'right'
            pad_length = pad_to_length - prefix_length_wo_pad
            if pad_length < 0:  # example too long
                continue
            # left padding: reverse, right-pad, reverse
            clean_prefix = torch.flip(F.pad(torch.flip(clean_prefix, (1,)), (0, pad_length), value=tokenizer.pad_token_id), (1,))

        # generate image embeddings
        pixel_values = img_processor(images=sample["image"].convert(mode="RGB"), return_tensors="pt")["pixel_values"]

        example_dict = {"clean_prefix": clean_prefix,
                            "clean_answer": clean_answer.item(),
                            "distractors": prepared_distractors,
                            "question_type": lookup[sample["question_type"]],
                            "prefix_length_wo_pad": prefix_length_wo_pad,
                            "pixel_values": pixel_values}
        examples.append(example_dict)
        if n_samples > 0 and len(examples) == n_samples:
            break
        
    return examples

def load_mmstar_examples(tokenizer, img_processor, pad_to_length, n_samples):
    samples = load_mmstar(n_samples)
    eos_token_id = tokenizer.convert_tokens_to_ids("</s>")
    examples = []
    sample_keys = list(samples.keys())
    random.seed(42)
    random.shuffle(sample_keys)
    correct_first_split = int(round(len(sample_keys)/2))
    for i, sample_key in enumerate(sample_keys):
        sample = samples[sample_key]
        random_prompt = random.choice(sample["prompts"][1:])

        if i < correct_first_split:
            part1 = sample["prompts"][0][0] + sample["prompts"][0][1]
            part2 = random_prompt[0] + random_prompt[1]
            
        else:
            part1 = random_prompt[0] + random_prompt[1]
            part2 = sample["prompts"][0][0] + sample["prompts"][0][1]
        text = part1 + "\n" + part2
        
        clean_prefix = tokenizer(text, return_tensors="pt",padding=False).input_ids

        tokens1 = tokenizer(part1, return_tensors="pt",padding=False).input_ids
        tokens1 = tokens1[tokens1 != eos_token_id]
        tokens2 = tokenizer(part2, return_tensors="pt",padding=False).input_ids
        tokens2 = tokens2[tokens2 != eos_token_id]

        answers_concat = torch.cat((tokens1, tokens2, torch.tensor([eos_token_id])))
        
        if not torch.equal(answers_concat, clean_prefix.flatten()):
            print("separate token encodings differ from combined token encoding")
            continue

        if i < correct_first_split:
            start_correct = 0
            end_correct = len(tokens1)
            start_incorrect = end_correct
            end_incorrect = start_incorrect + len(tokens2)
        else:
            start_incorrect = 0
            end_incorrect = len(tokens1)
            start_correct = end_incorrect
            end_correct = start_correct + len(tokens2)

        correct_idx = list(range(start_correct, end_correct))
        incorrect_idx = list(range(start_incorrect, end_incorrect))

        
        # if we specify `pad_to_length`, left-pad all inputs to a max length
        prefix_length_wo_pad = clean_prefix.shape[1]
        if pad_to_length:
            tokenizer.padding_side = 'right'
            pad_length = pad_to_length - prefix_length_wo_pad
            if pad_length < 0:  # example too long
                print(f"too long: {pad_length}")
                continue
            # left padding: reverse, right-pad, reverse
            clean_prefix = torch.flip(F.pad(torch.flip(clean_prefix, (1,)), (0, pad_length), value=tokenizer.pad_token_id), (1,))

        # generate image embeddings
        pixel_values = img_processor(images=sample["image"].convert(mode="RGB"), return_tensors="pt")["pixel_values"]

        example_dict = {"clean_prefix": clean_prefix,
                        "correct_idx": correct_idx,
                        "incorrect_idx": incorrect_idx,
                        "category": sample["category"],
                        "l2_category": sample["l2_category"],
                        "pixel_values": pixel_values,
                        "prefix_length_wo_pad": prefix_length_wo_pad}
        examples.append(example_dict)
        if n_samples > 0 and len(examples) == n_samples:
            break
    return examples

def load_ewok_examples(tokenizer, pad_to_length, n_samples):
    samples = load_ewok(n_samples, load_all=True)
    eos_token_id = tokenizer.convert_tokens_to_ids("</s>")
    examples = []
    sample_keys = list(samples.keys())
    random.seed(42)
    random.shuffle(sample_keys)
    correct_first_split = int(round(len(sample_keys)/2))
    for i, sample_key in enumerate(sample_keys):
        sample = samples[sample_key]

        if i < correct_first_split:
            part1 = sample["Target1"]
            part2 = sample["Target2"]
            
        else:
            part1 = sample["Target2"]
            part2 = sample["Target1"]
        text = sample["Context1"] + " " + part1 + "\n" + sample["Context1"] + " " + part2
        
        clean_prefix = tokenizer(text, return_tensors="pt",padding=False).input_ids

        tokens1 = tokenizer(part1, return_tensors="pt",padding=False).input_ids.flatten()
        tokens2 = tokenizer(part2, return_tensors="pt",padding=False).input_ids.flatten()
        context = tokenizer(sample["Context1"], return_tensors="pt",padding=False).input_ids.flatten()
        
        tokens1 = tokens1[tokens1 != eos_token_id]
        tokens2 = tokens2[tokens2 != eos_token_id]
        context = context[context != eos_token_id]
        
        answers_concat = torch.cat((context, tokens1, context, tokens2, torch.tensor([eos_token_id])))
        
        assert torch.equal(answers_concat, clean_prefix.flatten()), "separate token encodings differ from combined token encoding"

        if i < correct_first_split:
            start_correct = 0
            end_correct = len(context) + len(tokens1)
            start_incorrect = end_correct
            end_incorrect = start_incorrect + len(context) + len(tokens2)
        else:
            
            start_incorrect = 0
            end_incorrect = len(context) + len(tokens1)
            start_correct = end_incorrect
            end_correct = end_incorrect + len(context) + + len(tokens2)

        correct_idx = list(range(start_correct, end_correct))
        incorrect_idx = list(range(start_incorrect, end_incorrect))

        # if we specify `pad_to_length`, left-pad all inputs to a max length
        prefix_length_wo_pad = clean_prefix.shape[1]
        if pad_to_length:
            tokenizer.padding_side = 'right'
            pad_length = pad_to_length - prefix_length_wo_pad
            if pad_length < 0:  # example too long
                continue
            # left padding: reverse, right-pad, reverse
            clean_prefix = torch.flip(F.pad(torch.flip(clean_prefix, (1,)), (0, pad_length), value=tokenizer.pad_token_id), (1,))


        example_dict = {"clean_prefix": clean_prefix,
                        "correct_idx": correct_idx,
                        "incorrect_idx": incorrect_idx,
                        "Domain": sample["Domain"],
                        "prefix_length_wo_pad": prefix_length_wo_pad}
        examples.append(example_dict)
        if n_samples > 0 and len(examples) == n_samples:
            break
    return examples

def load_blimp_examples(tokenizer, pad_to_length, n_samples):
    samples = load_blimp(task="blimp_filtered", load_all=True)
    # identify suitable sentences for attribution patching
    suitable_samples = []
    for sample_id, sample in samples.items():
        if sample["one_prefix_method"] is True:
            # catch samples with errors (i.e. two same sentences)
            if sample["sentence_good"] == sample["sentence_bad"]:
                continue
            # sanity check
            sent_good = sample["sentence_good"].split()
            sent_bad = sample["sentence_bad"].split()
            if sent_good[:-1] == sent_bad[:-1]:
                sample["clean_answer"] = f" {sent_good[-1].rstrip(punctuation)}"
                sample["patch_answer"] = f" {sent_bad[-1].rstrip(punctuation)}"
                sample["clean_prefix"] = sample["sentence_good"].replace(sample["clean_answer"],"")
                suitable_samples.append(sample)
    examples = []
    eos_token_id = tokenizer.convert_tokens_to_ids("</s>")    
    for sample in suitable_samples:
        clean_tokens = tokenizer(sample["clean_prefix"], return_tensors="pt",
                                        padding=False)
        clean_prefix = clean_tokens.input_ids
        clean_answer = tokenizer(sample["clean_answer"], return_tensors="pt",
                                        padding=False).input_ids
        patch_answer = tokenizer(sample["patch_answer"], return_tensors="pt",
                                        padding=False).input_ids
        
        
        # remove EOS tokens from answers
        clean_answer = clean_answer[clean_answer != eos_token_id].unsqueeze(0)
        patch_answer = patch_answer[patch_answer != eos_token_id].unsqueeze(0)
        
        # only keep examples where answers are single tokens
        if clean_answer.shape[1] != 1 or patch_answer.shape[1] != 1:
            continue

        # if we specify `pad_to_length`, left-pad all inputs to a max length
        prefix_length_wo_pad = clean_prefix.shape[1]
        if pad_to_length:
            tokenizer.padding_side = 'right'
            pad_length = pad_to_length - prefix_length_wo_pad
            if pad_length < 0:  # example too long
                continue
            # left padding: reverse, right-pad, reverse
            clean_prefix = torch.flip(F.pad(torch.flip(clean_prefix, (1,)), (0, pad_length), value=tokenizer.pad_token_id), (1,))

        example_dict = {"clean_prefix": clean_prefix,
                            "clean_answer": clean_answer.item(),
                            "patch_answer": patch_answer.item(),
                            "UID": sample["UID"],
                            "linguistics_term": sample["linguistics_term"],
                            "prefix_length_wo_pad": prefix_length_wo_pad}
        examples.append(example_dict)
        if n_samples > 0 and len(examples) == n_samples:
            break
    return examples
        
def load_winoground_examples(tokenizer, img_processor, pad_to_length, n_samples):
    examples = []
    eos_token_id = tokenizer.convert_tokens_to_ids("</s>")
    winoground_data = load_vl_data(task="winoground", n_samples=n_samples)
    sample_keys = list(winoground_data.keys())
    random.seed(42)
    random.shuffle(sample_keys)
    correct_first_split = int(round(len(sample_keys)/2))
    lookup = parse_winoground_qtypes()
    for i, sample_key in enumerate(sample_keys):
        sample = winoground_data[sample_key]

        if i < correct_first_split:
            part1 = sample["caption_0"]
            part2 = sample["caption_1"]
            
        else:
            part1 = sample["caption_1"]
            part2 = sample["caption_0"]
        text = part1 + "\n" + part2
        
        clean_prefix = tokenizer(text, return_tensors="pt",padding=False).input_ids

        tokens1 = tokenizer(part1, return_tensors="pt",padding=False).input_ids
        tokens1 = tokens1[tokens1 != eos_token_id]
        tokens2 = tokenizer(part2, return_tensors="pt",padding=False).input_ids
        tokens2 = tokens2[tokens2 != eos_token_id]

        answers_concat = torch.cat((tokens1, tokens2, torch.tensor([eos_token_id])))
        
        assert torch.equal(answers_concat, clean_prefix.flatten()), "separate token encodings differ from combined token encoding"

        if i < correct_first_split:
            start_correct = 0
            end_correct = len(tokens1)
            start_incorrect = end_correct
            end_incorrect = start_incorrect + len(tokens2)
        else:
            start_incorrect = 0
            end_incorrect = len(tokens1)
            start_correct = end_incorrect
            end_correct = start_correct + len(tokens2)

        correct_idx = list(range(start_correct, end_correct))
        incorrect_idx = list(range(start_incorrect, end_incorrect))


        # if we specify `pad_to_length`, left-pad all inputs to a max length
        prefix_length_wo_pad = clean_prefix.shape[1]
        if pad_to_length:
            tokenizer.padding_side = 'right'
            pad_length = pad_to_length - prefix_length_wo_pad
            if pad_length < 0:  # example too long
                continue
            # left padding: reverse, right-pad, reverse
            clean_prefix = torch.flip(F.pad(torch.flip(clean_prefix, (1,)), (0, pad_length), value=tokenizer.pad_token_id), (1,))

        # generate image embeddings
        pixel_values = img_processor(images=sample["image"].convert(mode="RGB"), return_tensors="pt")["pixel_values"]

        example_dict = {"clean_prefix": clean_prefix,
                        "correct_idx": correct_idx,
                        "incorrect_idx": incorrect_idx,
                        "superclass": lookup[sample["tag"]],
                        "tag": sample["tag"],
                        "collapsed_tag": sample["collapsed_tag"],
                        "secondary_tag": sample["secondary_tag"],
                        "pixel_values": pixel_values,
                        "prefix_length_wo_pad": prefix_length_wo_pad}
        examples.append(example_dict)
        if n_samples > 0 and len(examples) == n_samples:
            break
    return examples

def parse_vqa_qtypes():
    with open("vqa_superclasses.txt", "r") as f:
        lines = f.readlines()
        mapping = {}
        for l in lines:
            parts = l.split("-")
            mapping[parts[0].strip()] = parts[1].strip()
    return mapping

def parse_winoground_qtypes():
    with open("winoground_superclasses.txt", "r") as f:
        lines = f.readlines()
        mapping = {}
        for l in lines:
            parts = l.split(":")
            mapping[parts[0].strip()] = parts[1].strip()
    return mapping

def create_nnsight(model):
    """create nnsight wrapper for transformer model"""
    
    nnsight_model = NNsight(model, device_map="cuda")
    nnsight_model.to("cuda")

    return nnsight_model

def load_git_model(model_dir, model_setting, epoch):
    model_path = f"{model_dir}/base_{model_setting}_e{epoch}/"
    spec = importlib.util.spec_from_file_location("GitForCausalLM", f"{model_path}modeling_git.py")
    git_module = importlib.util.module_from_spec(spec)
    sys.modules["git_module"] = git_module
    spec.loader.exec_module(git_module)
    GitForCausalLM = git_module.GitForCausalLM

    model = GitForCausalLM.from_pretrained(model_path) 
    ckpt = torch.load(model_path + "pytorch_model.bin") # TODO: newly initialized for vision encoder: ['pooler.dense.bias', 'pooler.dense.weight']
    model.load_state_dict(ckpt, strict=False)  
        
    # load tokenizer and img processor
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if model_setting.startswith("git"):
        print("loading processor")
        img_processor = AutoProcessor.from_pretrained(
                            model_path,
                            trust_remote_code=True
                        )
    else:
        print("not loading processor")
        img_processor = None

    model.to("cuda")
    return model, img_processor, tokenizer


def load_flamingo_model():
    model_path = "babylm/flamingo-2024"

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    img_processor = AutoProcessor.from_pretrained(model_path,trust_remote_code=True)
    device = torch.device("cuda")
    model.to(device)

    return model, img_processor, tokenizer