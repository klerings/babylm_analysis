import torch
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from PIL import Image
import gc
import sys
from transformers import AutoProcessor, AutoTokenizer
import importlib.util
from loading_utils import load_blimp, load_mmstar, load_vl_data, load_flamingo_model, load_git_model

 
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
    subset_size = -1 # use all samples
    batch_size = 64

    txt_only_tasks = ["blimp_filtered", "supplement_filtered"]
    vl_tasks = ["mmstar", "vqa", "winoground"]
    tasks = vl_tasks + txt_only_tasks

    # choose specific GIT model or pretrained Flamingo model
    #settings = ["git_1vd25_s1"]
    settings = ["flamingo"]

    model_dir = "../babylm_GIT/models_for_eval/final_models" # must be changed depending on where trained models are stored

    for setting in settings:
        if setting == "flamingo":
            epoch = None
            model, image_processor, tokenizer = load_flamingo_model()
        else:
            epoch = 29
            model, image_processor, tokenizer = load_git_model(model_dir, setting, epoch)
        
        
        if model is None:
            print(f"skipping: {setting} - {epoch}")
            continue

        for task in tasks:

            if task in txt_only_tasks:
                samples = load_blimp(task, n_samples=subset_size)
            elif task == "mmstar":
                samples = load_mmstar(n_samples=subset_size)
            else:
                samples = load_vl_data(task, n_samples=subset_size)
            
            acc_no_img = run_eval(samples, model, image_processor, tokenizer, mode="txt_only", task=task, batch_size=batch_size)
            print(f"\n{setting} - {epoch} - {task}")
            print(f"-> no img: {acc_no_img}")

            torch.cuda.empty_cache()
            gc.collect()

            if (setting.startswith("git") or setting == "flamingo") and task not in txt_only_tasks:

                acc_with_img = run_eval(samples, model, image_processor, tokenizer, mode="with_img",  task=task, batch_size=batch_size)
                print(f"\n{setting} - {epoch} - {task}")
                print(f"-> with img: {acc_with_img}")
                
                torch.cuda.empty_cache()
                gc.collect()

                #acc_with_noise = run_eval(samples, model, image_processor, tokenizer, mode="with_noise",  task=task, batch_size=batch_size)
                #print(f"-> with noise: {acc_with_noise}")

                #torch.cuda.empty_cache()
                #gc.collect()
        



    
