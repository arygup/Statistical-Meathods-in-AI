from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")
print(model)

from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing import Any, Dict, List, Optional, Union
import torch

@dataclass
class DataCollatorForCausalLM:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = [feature['input_ids'] for feature in features]
        labels = [feature['labels'] for feature in features]
        batch_input = self.tokenizer.pad(
            {'input_ids': input_ids},
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )
        batch_labels = self.tokenizer.pad(
            {'input_ids': labels},
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )
        batch_labels['input_ids'][batch_labels['input_ids'] == self.tokenizer.pad_token_id] = -100
        batch = {
            'input_ids': batch_input['input_ids'],
            'attention_mask': batch_input['attention_mask'],
            'labels': batch_labels['input_ids'],
        }

        return batch


from datasets import load_dataset
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding, TrainingArguments
import numpy as np
import evaluate
from trl import SFTTrainer
import torch
import torch.nn as nn
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing import Any, Dict, List, Optional, Union
import torch

rouge = evaluate.load("rouge")
data_files = {"train": "train.csv", "test": "test.csv", "validation": "validation.csv"}
dataset = load_dataset("/kaggle/input/newspaper-text-summarization-cnn-dailymail/cnn_dailymail", data_files=data_files)
num_train_samples = 27000
train_select = random.sample(range(len(dataset["train"])), k=num_train_samples)

num_validation_samples = 100
validation_select = random.sample(range(len(dataset["validation"])), k=num_validation_samples)

num_test_samples = 100
test_select = random.sample(range(len(dataset["test"])), k=num_test_samples)

dataset_train = dataset["train"].select(train_select)
dataset_validation = dataset["validation"].select(validation_select)
dataset_test = dataset["test"].select(test_select)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

num_soft_tokens = 20  
embedding_size = 768  

class GPT2WithSoftPrompt(nn.Module):
    def __init__(self, base_model, num_soft_tokens):
        super().__init__()
        self.base_model = base_model
        self.num_soft_tokens = num_soft_tokens
        self.soft_prompt_embeddings = nn.Parameter(torch.randn(num_soft_tokens, base_model.config.hidden_size))
        for param in self.base_model.parameters():
            param.requires_grad = False

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        input_embeds = self.base_model.transformer.wte(input_ids)
        batch_size = input_ids.size(0)
        soft_prompt_embeds = self.soft_prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        inputs_embeds = torch.cat([soft_prompt_embeds, input_embeds], dim=1)
        if attention_mask is not None:
            soft_prompt_mask = torch.ones(batch_size, self.num_soft_tokens, dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([soft_prompt_mask, attention_mask], dim=1)
        if labels is not None:
            soft_prompt_labels = torch.full((batch_size, self.num_soft_tokens), -100, dtype=labels.dtype, device=labels.device)
            labels = torch.cat([soft_prompt_labels, labels], dim=1)
        position_ids = torch.arange(inputs_embeds.size(1), dtype=torch.long, device=inputs_embeds.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        outputs = self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids,
        )
        return outputs

base_model = AutoModelForCausalLM.from_pretrained("gpt2")
model = GPT2WithSoftPrompt(base_model, num_soft_tokens=num_soft_tokens)

def preprocess_function(examples):
    inputs = examples["article"]
    model_inputs = tokenizer(inputs, max_length=400, truncation=True, padding='max_length')
    labels = tokenizer(text_target=examples["highlights"], max_length=400, truncation=True, padding='max_length')
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = dataset_train.map(preprocess_function, batched=True)
tokenized_validation = dataset_validation.map(preprocess_function, batched=True)
tokenized_test = dataset_test.map(preprocess_function, batched=True)

def compute_metrics(eval_pred):
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    predictions = predictions[:, num_soft_tokens:, :]
    labels = labels[:, num_soft_tokens:]
    predictions = np.argmax(predictions, axis=-1)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    return {k: round(v, 4) for k, v in result.items()}



training_args = TrainingArguments(
    output_dir="gpt2_soft_prompt",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=15,
    per_device_eval_batch_size=1,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=20,
    report_to="none",
    fp16=True,
    eval_accumulation_steps=1,
    save_safetensors=False,
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_train, 
    eval_dataset=tokenized_validation,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

trainer.train()
