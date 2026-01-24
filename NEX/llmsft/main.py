import torch
import numpy as np
import pandas as pd
from vllm import LLM, SamplingParams
from sklearn.metrics import f1_score
import os
import time
import json
import argparse
import subprocess
import sys

model_path = '{}'.format(sys.argv[1])

vLLM = LLM(model=model_path, tensor_parallel_size=int(sys.argv[4]), dtype="half", enforce_eager=True)

sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=1024, logprobs=1)


inference_set = []
prompts = []
with open('{}'.format(sys.argv[2])) as f:
    data_json = json.load(f)


truthes = []

for prompt in data_json:
    prompt_text = prompt["instruction"]
    prompts.append("[INST] %s [/INST]" % prompt_text)
    

outputs = vLLM.generate(prompts, sampling_params)

generation_list = []
for i in range(len(outputs)):
    prompt = data_json[i]
    output = outputs[i]
    generated_text = {
        'instruction': prompt["instruction"], 
        'key': prompt["key"], 
        'predict': output.outputs[0].text
    }
    generation_list.append(generated_text)
    

with open('{}_filled.json'.format(sys.argv[3]), 'w', encoding='utf-8') as file:
    json.dump(generation_list, file, ensure_ascii=False, indent=4)

