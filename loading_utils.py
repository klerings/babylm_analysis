import os
import re
import json
import random
import torch as t
import torch.nn.functional as F
from dictionary_learning.dictionary import AutoEncoder
from dataclasses import dataclass
from evaluate_tasks import load_vl_data, load_blimp
from string import punctuation

@dataclass
class DictionaryCfg():
    def __init__(
        self,
        dictionary_dir,
        dictionary_size
        ) -> None:
        self.dir = dictionary_dir
        self.size = dictionary_size


def load_examples(dataset, num_examples, model, seed=12, pad_to_length=None, length=None,
                  ignore_patch=False):
    examples = []
    dataset_items = open(dataset).readlines()
    random.seed(seed)
    random.shuffle(dataset_items)
    for line in dataset_items:
        data = json.loads(line)
        clean_prefix = model.tokenizer(data["clean_prefix"], return_tensors="pt",
                                        padding=False).input_ids
        patch_prefix = model.tokenizer(data["patch_prefix"], return_tensors="pt",
                                        padding=False).input_ids
        clean_answer = model.tokenizer(data["clean_answer"], return_tensors="pt",
                                        padding=False).input_ids
        patch_answer = model.tokenizer(data["patch_answer"], return_tensors="pt",
                                        padding=False).input_ids
        # remove BOS tokens from answers
        clean_answer = clean_answer[clean_answer != model.tokenizer.bos_token_id].unsqueeze(0)
        patch_answer = patch_answer[patch_answer != model.tokenizer.bos_token_id].unsqueeze(0)
        # only keep examples where answers are single tokens
        if not ignore_patch:
            if clean_prefix.shape[1] != patch_prefix.shape[1]:
                continue
        # only keep examples where clean and patch inputs are the same length
        if clean_answer.shape[1] != 1 or patch_answer.shape[1] != 1:
            continue
        # if we specify a `length`, filter examples if they don't match
        if length and clean_prefix.shape[1] != length:
            continue
        # if we specify `pad_to_length`, left-pad all inputs to a max length
        prefix_length_wo_pad = clean_prefix.shape[1]
        if pad_to_length:
            model.tokenizer.padding_side = 'right'
            pad_length = pad_to_length - prefix_length_wo_pad
            if pad_length < 0:  # example too long
                continue
            # left padding: reverse, right-pad, reverse
            clean_prefix = t.flip(F.pad(t.flip(clean_prefix, (1,)), (0, pad_length), value=model.tokenizer.pad_token_id), (1,))
            patch_prefix = t.flip(F.pad(t.flip(patch_prefix, (1,)), (0, pad_length), value=model.tokenizer.pad_token_id), (1,))
        example_dict = {"clean_prefix": clean_prefix,
                        "patch_prefix": patch_prefix,
                        "clean_answer": clean_answer.item(),
                        "patch_answer": patch_answer.item(),
                        "annotations": get_annotation(dataset, model, data),
                        "prefix_length_wo_pad": prefix_length_wo_pad,}
        examples.append(example_dict)
        if len(examples) >= num_examples:
            break

    return examples


def load_vqa_examples(tokenizer, img_processor, pad_to_length, n_samples, local):
    samples = load_vl_data(task="vqa", n_samples=n_samples*10, local=local)
    examples = []
    eos_token_id = tokenizer.convert_tokens_to_ids("</s>")
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
            clean_prefix = t.flip(F.pad(t.flip(clean_prefix, (1,)), (0, pad_length), value=tokenizer.pad_token_id), (1,))

        # generate image embeddings
        pixel_values = img_processor(images=sample["image"].convert(mode="RGB"), return_tensors="pt")["pixel_values"]

        example_dict = {"clean_prefix": clean_prefix,
                            "clean_answer": clean_answer.item(),
                            "distractors": prepared_distractors,
                            "question_type": sample["question_type"],
                            "prefix_length_wo_pad": prefix_length_wo_pad,
                            "pixel_values": pixel_values}
        examples.append(example_dict)
        if n_samples > 0 and len(examples) == n_samples:
            break
        
    return examples

def load_blimp_examples(tokenizer, pad_to_length, n_samples, local):
    samples = load_blimp(task="blimp_filtered", load_all=True, local=local)
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
                sample["patch_prefix"] = sample["sentence_bad"].replace(sample["patch_answer"],"")
                suitable_samples.append(sample)
    examples = []
    eos_token_id = tokenizer.convert_tokens_to_ids("</s>")    
    for sample in suitable_samples:
        clean_tokens = tokenizer(sample["clean_prefix"], return_tensors="pt",
                                        padding=False)
        clean_prefix = clean_tokens.input_ids
        patch_tokens = tokenizer(sample["patch_prefix"], return_tensors="pt",
                                        padding=False)
        patch_prefix = patch_tokens.input_ids
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

        # only keep examples where prefix is of same length
        if clean_prefix.shape[1] != patch_prefix.shape[1]:
            continue

        # if we specify `pad_to_length`, left-pad all inputs to a max length
        prefix_length_wo_pad = clean_prefix.shape[1]
        if pad_to_length:
            tokenizer.padding_side = 'right'
            pad_length = pad_to_length - prefix_length_wo_pad
            if pad_length < 0:  # example too long
                continue
            # left padding: reverse, right-pad, reverse
            clean_prefix = t.flip(F.pad(t.flip(clean_prefix, (1,)), (0, pad_length), value=tokenizer.pad_token_id), (1,))
            # left padding: reverse, right-pad, reverse
            patch_prefix = t.flip(F.pad(t.flip(patch_prefix, (1,)), (0, pad_length), value=tokenizer.pad_token_id), (1,))

        example_dict = {"clean_prefix": clean_prefix,
                            "clean_answer": clean_answer.item(),
                            "patch_prefix": patch_prefix,
                            "patch_answer": patch_answer.item(),
                            "UID": sample["UID"],
                            "linguistics_term": sample["linguistics_term"],
                            "prefix_length_wo_pad": prefix_length_wo_pad}
        examples.append(example_dict)
        if n_samples > 0 and len(examples) == n_samples:
            break
    return examples
        
def load_winoground_examples(tokenizer, img_processor, pad_to_length, n_samples, local):
    examples = []
    eos_token_id = tokenizer.convert_tokens_to_ids("</s>")
    winoground_data = load_vl_data(task="winoground", n_samples=n_samples, local=local)
    sample_keys = list(winoground_data.keys())
    random.seed(42)
    random.shuffle(sample_keys)
    correct_first_split = int(round(len(sample_keys)/2))
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

        answers_concat = t.cat((tokens1, tokens2, t.tensor([eos_token_id])))
        
        assert t.equal(answers_concat, clean_prefix.flatten()), "separate token encodings differ from combined token encoding"

        if i < correct_first_split:
            correct_idx = list(range(0, len(tokens1)))
            incorrect_idx = list(range(len(tokens1), len(tokens1)+len(tokens2)))
        else:
            correct_idx = list(range(0, len(tokens2)))
            incorrect_idx = list(range(len(tokens2), len(tokens2)+len(tokens1)))


        # if we specify `pad_to_length`, left-pad all inputs to a max length
        prefix_length_wo_pad = clean_prefix.shape[1]
        if pad_to_length:
            tokenizer.padding_side = 'right'
            pad_length = pad_to_length - prefix_length_wo_pad
            if pad_length < 0:  # example too long
                continue
            # left padding: reverse, right-pad, reverse
            clean_prefix = t.flip(F.pad(t.flip(clean_prefix, (1,)), (0, pad_length), value=tokenizer.pad_token_id), (1,))

        # generate image embeddings
        pixel_values = img_processor(images=sample["image"].convert(mode="RGB"), return_tensors="pt")["pixel_values"]

        example_dict = {"clean_prefix": clean_prefix,
                        "correct_idx": correct_idx,
                        "incorrect_idx": incorrect_idx,
                        "tag": sample["tag"],
                        "collapsed_tag": sample["collapsed_tag"],
                        "secondary_tag": sample["secondary_tag"],
                        "pixel_values": pixel_values,
                        "prefix_length_wo_pad": prefix_length_wo_pad}
        examples.append(example_dict)
        if n_samples > 0 and len(examples) == n_samples:
            break
    return examples

def load_examples_nopair(dataset, num_examples, model, length=None):
    examples = []
    if isinstance(dataset, str):        # is a path to a .json file
        dataset = json.load(open(dataset))
    elif isinstance(dataset, dict):     # is an already-loaded dictionary
        pass
    else:
        raise ValueError(f"`dataset` is unrecognized type: {type(dataset)}. Must be path (str) or dict")
    
    max_len = 0     # for padding
    for context_id in dataset:
        context = dataset[context_id]["context"]
        if length is not None and len(context) > length:
            context = context[-length:]
        clean_prefix = model.tokenizer("".join(context), return_tensors="pt",
                        padding=False).input_ids
        max_len = max(max_len, clean_prefix.shape[-1])

    for context_id in dataset:
        answer = dataset[context_id]["answer"]
        context = dataset[context_id]["context"]
        clean_prefix = model.tokenizer("".join(context), return_tensors="pt",
                                    padding=False).input_ids
        clean_answer = model.tokenizer(answer, return_tensors="pt",
                                    padding=False).input_ids
        if clean_answer.shape[1] != 1:
            continue
        prefix_length_wo_pad = clean_prefix.shape[1]
        pad_length = max_len - prefix_length_wo_pad
        # left padding: reverse, right-pad, reverse
        clean_prefix = t.flip(F.pad(t.flip(clean_prefix, (1,)), (0, pad_length), value=model.tokenizer.pad_token_id), (1,))

        example_dict = {"clean_prefix": clean_prefix,
                        "clean_answer": clean_answer.item(),
                        "prefix_length_wo_pad": prefix_length_wo_pad,}
        examples.append(example_dict)
        if len(examples) >= num_examples:
            break

    return examples

def get_annotation(dataset, model, data):
    # First, understand which dataset we're working with
    structure = None
    if "within_rc" in dataset:
        structure = "within_rc"
        template = "the_subj subj_main that the_dist subj_dist"
    elif "rc.json" in dataset or "rc_" in dataset:
        structure = "rc"
        template = "the_subj subj_main that the_dist subj_dist verb_dist"
    elif "simple.json" in dataset or "simple_" in dataset:
        structure = "simple"
        template = "the_subj subj_main"
    elif "nounpp.json" in dataset or "nounpp_" in dataset:
        structure = "nounpp"
        template = "the_subj subj_main prep the_dist subj_dist"

    if structure is None:
        return {}
    
    annotations = {}

    # Iterate through words in the template and input. Get token spans
    curr_token = 0
    for template_word, word in zip(template.split(), data["clean_prefix"].split()):
        if word != "The":
            word = " " + word
        word_tok = model.tokenizer(word, return_tensors="pt", padding=False).input_ids
        num_tokens = word_tok.shape[1]
        span = (curr_token, curr_token + num_tokens-1)
        curr_token += num_tokens
        annotations[template_word] = span
    
    return annotations