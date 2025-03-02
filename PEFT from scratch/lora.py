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
import numpy as np
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

rouge = evaluate.load("rouge")
dataset = load_dataset("cnn_dailymail", "3.0.0")
num_train_samples = 27000
train_select = random.sample(range(len(dataset["train"])), k=num_train_samples)
num_validation_samples = 100
validation_select = random.sample(range(len(dataset["validation"])), k=num_validation_samples)

num_test_samples = 100
test_select = random.sample(range(len(dataset["test"])), k=num_test_samples)

dataset_train = dataset["train"].select(train_select)
dataset_validation = dataset["validation"].select(validation_select)
dataset_test = dataset["test"].select(test_select)

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def preprocess_function(examples):
    prefix = "summarize: "
    inputs = [prefix + doc for doc in examples["article"]]
    targets = [ex for ex in examples["highlights"]]
    max_source_length = 512 
    max_target_length = 128  
    tokenized_inputs = tokenizer(
        inputs, max_length=max_source_length, truncation=True, padding=False
    )
    tokenized_targets = tokenizer(
        targets, max_length=max_target_length, truncation=True, padding=False
    )
    input_ids = []
    labels = []
    for i in range(len(tokenized_inputs["input_ids"])):
        input_ids_input = tokenized_inputs["input_ids"][i]
        input_ids_target = tokenized_targets["input_ids"][i]
        input_ids_combined = input_ids_input + input_ids_target
        input_ids_combined = input_ids_combined[:1024]  
        input_ids.append(input_ids_combined)
        labels_combined = [-100] * len(input_ids_input) + input_ids_target
        labels_combined = labels_combined[:1024]  
        labels.append(labels_combined)
    model_inputs = {
        "input_ids": input_ids,
        "labels": labels
    }
    return model_inputs

tokenized_train = dataset_train.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

tokenized_validation = dataset_validation.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["validation"].column_names,
)

tokenized_test = dataset_test.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["test"].column_names,
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["c_attn"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions
    ]
    result["gen_len"] = np.mean(prediction_lens)
    return {k: round(v, 4) for k, v in result.items()}

data_collator = DataCollatorForCausalLM(tokenizer=tokenizer, padding='longest')

training_args = TrainingArguments(
    output_dir="gpt2_lora_test",
    evaluation_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=2,  
    per_device_eval_batch_size=2,   
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=10,
    report_to="none",
    logging_steps=10,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    eval_accumulation_steps=2
)

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

