batch = [winoground_examples[i:min(i + 2, 10)] for i in range(0, 10, 2)][0]
clean_inputs = t.cat([e['clean_prefix'] for e in batch], dim=0).to("cuda")
correct_idxs = [[0,1,2], [4,5,6,7]]
incorrect_idxs = [[1,2], [5,6,7]]

"""
logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1 )  # [1, seq]
                answer = float(logits.sum())
"""
tracer_kwargs = {'validate' : False, 'scan' : False}
with model.trace(**tracer_kwargs) as tracer:
            
    with tracer.invoke(clean_inputs, scan=tracer_kwargs['scan']):
        outputs = model.output.output.save()


a = t.gather(outputs[:,-1,:], dim=-1, index=t.tensor([17, 18]).to("cuda").view(-1, 1)).squeeze(-1)
b = t.gather(outputs[:,-1,:], dim=-1, index=t.tensor([19, 13]).to("cuda").view(-1, 1)).squeeze(-1)
print(a)
print(b)
print(a-b)

correct_sent_logits = []
incorrect_sent_logits = []
for i, idx in enumerate(correct_idxs):
    use = t.tensor([idx]).to("cuda")
    logits = torch.gather(outputs[i,:,:], dim=1, index=use).squeeze(-1) # [1, seq]
    print(logits.sum())
    print(logits.sum().unsqueeze(0).shape)
    correct_sent_logits.append(logits.sum().unsqueeze(0))
correct_sent_logits = torch.cat(correct_sent_logits, dim=0)
print(correct_sent_logits-correct_sent_logits)

# output
"""
{'clean_prefix': tensor([[   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
             0,  310,  401,  114, 7434,   45,    5,    1]]),
 'clean_answer': 1370,
 'patch_prefix': tensor([[ 310,  401,  114, 7434,   45,    5,    1]]),
 'patch_answer': 404,
 'UID': 'anaphor_gender_agreement',
 'linguistics_term': 'anaphor_agreement',
 'prefix_length_wo_pad': 7}
/home/alina/miniconda3/envs/babylm/lib/python3.9/site-packages/datasets/load.py:1461: FutureWarning: The repository for HuggingFaceM4/VQAv2 contains custom code which must be executed to correctly load the dataset. You can inspect the repository content at https://hf.co/datasets/HuggingFaceM4/VQAv2
You can avoid this message in future by passing the argument `trust_remote_code=True`.
Passing `trust_remote_code=True` will be mandatory to load this dataset from the next major release of `datasets`.
  warnings.warn(
Repo card metadata block was not found. Setting CardData to empty.
loaded huggingface DS
loaded local DS
100%|██████████| 100/100 [00:00<00:00, 185.25it/s]
{'clean_prefix': tensor([[   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
             0,   27,   44, 4045,   23,  463,   17,    1]]),
 'clean_answer': 49,
 'distractors': [3895, 1224, 121, 1017, 303, 55, 175],
 'question_type': 'is this',
 'prefix_length_wo_pad': 7,
 'pixel_values': tensor([[[[-2.1179, -2.1179, -2.1179,  ..., -2.1179, -2.1179, -2.1179],
           [-2.1179, -2.1179, -2.1179,  ..., -2.1008, -2.1179, -2.1179],
           [-2.1179, -2.1008, -2.1179,  ..., -2.1008, -2.1179, -2.1179],
           ...,
           [-2.1008, -2.1179, -2.1008,  ..., -2.1179, -2.1179, -2.1179],
           [-2.1008, -2.1179, -2.1008,  ..., -2.1179, -2.1179, -2.1179],
           [-2.1179, -2.1179, -2.1179,  ..., -2.1179, -2.1179, -2.1179]],
 
          [[-2.0357, -2.0357, -2.0357,  ..., -2.0357, -2.0357, -2.0357],
           [-2.0357, -2.0357, -2.0357,  ..., -2.0182, -2.0357, -2.0357],
           [-2.0357, -2.0182, -2.0357,  ..., -2.0182, -2.0357, -2.0357],
           ...,
           [-2.0182, -2.0357, -2.0182,  ..., -2.0357, -2.0357, -2.0357],
           [-2.0182, -2.0357, -2.0182,  ..., -2.0357, -2.0357, -2.0357],
           [-2.0357, -2.0357, -2.0357,  ..., -2.0357, -2.0357, -2.0357]],
 
          [[-1.8044, -1.8044, -1.8044,  ..., -1.7870, -1.8044, -1.8044],
           [-1.8044, -1.8044, -1.8044,  ..., -1.7870, -1.8044, -1.8044],
           [-1.8044, -1.7870, -1.7870,  ..., -1.7870, -1.8044, -1.8044],
           ...,
           [-1.7870, -1.8044, -1.7870,  ..., -1.8044, -1.8044, -1.8044],
           [-1.7870, -1.8044, -1.7870,  ..., -1.8044, -1.8044, -1.8044],
           [-1.8044, -1.8044, -1.8044,  ..., -1.8044, -1.8044, -1.7870]]]])}
tensor([-0.7798,  0.7615], device='cuda:0', grad_fn=<SqueezeBackward1>)
tensor([-1.1319, -3.8153], device='cuda:0', grad_fn=<SqueezeBackward1>)
tensor([0.3521, 4.5768], device='cuda:0', grad_fn=<SubBackward0>)
"""

