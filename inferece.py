import sys
import json
from alpaca_eval.decoders.huggingface_local import huggingface_local_completions

"""
model_name = "/workspace/mnt/model_data/share/model/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/8c22764a7e3675c50d4c7c9a4edb474456022b16"
model_name = "/workspace/mnt/model_data/share/model/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9"
model_name = "/root/mark/llama-1/alpaca"
model_name = "outputs"
model_name = "outputs_hf"
model_name = "outputs_trl"
"""

model_name = sys.argv[1]

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{response}"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n{response}"
    ),
}

prompts = []

# outputs.json IS FROM alpaca_eval
with open('outputs.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for d in data:
    prompt_no_input = PROMPT_DICT['prompt_no_input']
    prompt_no_input = prompt_no_input.format_map({
        "instruction": d['instruction'],
        "response": ""
    })
    prompts.append(prompt_no_input)

results = huggingface_local_completions(
    prompts=prompts,
    model_name=model_name,
    do_sample=False,
    batch_size=8,
)

with open(f'outputs_results-{model_name}.json'.replace('/', '_'), 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=True, indent=4)

for i in range(len(data)):
    data[i]['output'] = results['completions'][i].strip()

with open(f'outputs_results_format-{model_name}.json'.replace('/', '_'), 'w', encoding='utf-8') as f:
    json.dump(data, f,  ensure_ascii=True, indent=4)

