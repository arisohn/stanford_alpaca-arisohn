from transformers import Trainer
from trl import SFTTrainer
from pprint import pprint

import os
import torch

def _print(str):
    with open('output.log', 'a', encoding='utf-8') as f:
        print(str, file=f, flush=True)        

def _pprint(str):
    with open('output.log', 'a', encoding='utf-8') as f:
        pprint(str, stream=f)
        f.flush()


class CustomTrainer(Trainer):
    def training_step(self, *args, **kwargs):
        if self.state.global_step == 0 and  int(os.environ['RANK']) == 0:
            inputs = args[1] if len(args) > 1 else kwargs.get('inputs', None)

            input_ids = inputs.get('input_ids', None)
            labels = inputs.get('labels', None)
            attention_mask = inputs.get('attention_mask', None)
            _print("-" * 100)
            _print(f"[CustomTrainer] input_id: {input_ids[0]}")
            _print(f"[CustomTrainer] label: {labels[0]}")
            _print(f"[CustomTrainer] attention_mask: {attention_mask[0]}")

            decoded = self.processing_class.convert_ids_to_tokens(input_ids[0])
            _print(f"[CustomTrainer] input_id decoded: {decoded}")
            decoded = self.processing_class.decode(input_ids[0], skip_special_tokens=False)
            _print(f"[CustomTrainer] input_id decoded: {decoded}")

            label = torch.where(labels[0] == -100, torch.tensor(self.processing_class.pad_token_id, device=labels.device), labels[0])
            decoded = self.processing_class.convert_ids_to_tokens(label)
            _print(f"[CustomTrainer] label decoded: {decoded}")
            decoded = self.processing_class.decode(label, skip_special_tokens=False)
            _print(f"[CustomTrainer] label decoded: {decoded}")

        return super().training_step(*args, **kwargs)


class CustomSFTTrainer(SFTTrainer):
    def training_step(self, *args, **kwargs):
        if self.state.global_step == 0 and  int(os.environ['RANK']) == 0:
            inputs = args[1] if len(args) > 1 else kwargs.get('inputs', None)

            input_ids = inputs.get('input_ids', None)
            labels = inputs.get('labels', None)
            attention_mask = inputs.get('attention_mask', None)
            _print("-" * 100)
            _print(f"[CustomSFTTrainer] input_id: {input_ids[0]}")
            _print(f"[CustomSFTTrainer] label: {labels[0]}")
            _print(f"[CustomSFTTrainer] attention_mask: {attention_mask[0]}")
            
            decoded = self.processing_class.convert_ids_to_tokens(input_ids[0])
            _print(f"[CustomSFTTrainer] input_id decoded: {decoded}")
            decoded = self.processing_class.decode(input_ids[0], skip_special_tokens=False)
            _print(f"[CustomSFTTrainer] input_id decoded: {decoded}")

            label = torch.where(labels[0] == -100, torch.tensor(self.processing_class.pad_token_id, device=labels.device), labels[0])
            decoded = self.processing_class.convert_ids_to_tokens(label)
            _print(f"[CustomSFTTrainer] label decoded: {decoded}")
            decoded = self.processing_class.decode(label, skip_special_tokens=False)
            _print(f"[CustomSFTTrainer] label decoded: {decoded}")

        return super().training_step(*args, **kwargs)

