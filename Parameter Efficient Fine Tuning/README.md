# GPT-2 Text Summarization: Fine-Tuning with Standard, Soft Prompts, and LoRA Methods

We explore various methods to fine-tune the GPT-2 language model for text summarization using the CNN/DailyMail dataset. We implement and compare three approaches:

- **Standard Fine-Tuning**: Adjusting the model's weights through traditional fine-tuning.
- **Soft Prompt Tuning**: Introducing trainable prompts while keeping the model's original weights frozen.
- **LoRA (Low-Rank Adaptation)**: Efficient fine-tuning by injecting low-rank matrices into the model's layers.

---

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Methods](#methods)
  - [Standard Fine-Tuning](#standard-fine-tuning)
  - [Soft Prompt Tuning](#soft-prompt-tuning)
  - [LoRA Fine-Tuning](#lora-fine-tuning)
- [Implementation](#implementation)
  - [Data Preprocessing](#data-preprocessing)
  - [Fine-Tuning (finetuning.py)](#fine-tuning-finetuningpy)
  - [Soft Prompt Tuning (prompt.py)](#soft-prompt-tuning-promptpy)
  - [LoRA Fine-Tuning (lora.py)](#lora-fine-tuning-lorapy)
- [Results](#results)
- [Conclusion](#conclusion)
- [Refrences](#references)

---

## Introduction

Text summarization condenses lengthy text into concise summaries, capturing the main ideas. With large language models like GPT-2, we can perform abstractive summarization by generating summaries in natural language. This project investigates different fine-tuning strategies to optimize GPT-2 for summarization tasks, balancing performance and computational efficiency.

---

## Dataset

We utilize the **CNN/DailyMail** dataset, a benchmark for summarization tasks containing news articles and their corresponding highlights (summaries).

---

## Requirements

- Python 3.x
- Transformers
- Datasets
- Evaluate
- TRL (Transformer Reinforcement Learning)
- PEFT (Parameter-Efficient Fine-Tuning)
- PyTorch
- NumPy

Install the dependencies using:

```bash
pip install transformers datasets evaluate trl peft torch numpy
```

---

## Methods

### Standard Fine-Tuning

Standard fine-tuning involves training the entire GPT-2 model or parts of it on the summarization dataset, adjusting the model's weights to better fit the new task.

### Soft Prompt Tuning

Soft prompt tuning freezes the original model weights and introduces trainable soft prompts. These prompts are additional tokens whose embeddings are learned during training, guiding the model without altering its pre-trained parameters.

### LoRA Fine-Tuning

LoRA (Low-Rank Adaptation) reduces the number of trainable parameters by injecting low-rank decomposition matrices into the model's layers. This method is parameter-efficient and accelerates training.

---

## Implementation

### Data Preprocessing

#### Loading the Dataset

```python
from datasets import load_dataset
import random

# Load the CNN/DailyMail dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")

# Sample subsets for training, validation, and testing
num_train_samples = 27000
train_select = random.sample(range(len(dataset["train"])), k=num_train_samples)

num_validation_samples = 100
validation_select = random.sample(range(len(dataset["validation"])), k=num_validation_samples)

num_test_samples = 100
test_select = random.sample(range(len(dataset["test"])), k=num_test_samples)

# Create the subsets
dataset_train = dataset["train"].select(train_select)
dataset_validation = dataset["validation"].select(validation_select)
dataset_test = dataset["test"].select(test_select)
```

#### Tokenization and Preprocessing

A common preprocessing function is used to tokenize inputs and prepare labels:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def preprocess_function(examples):
    prefix = "summarize: "
    inputs = [prefix + doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding='max_length')
    labels = tokenizer(text_target=examples["highlights"], max_length=128, truncation=True, padding='max_length')
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
```

---

### Fine-Tuning (finetuning.py)

#### Loading the Model and Tokenizer

```python
from transformers import AutoModelForCausalLM

# Load the GPT-2 model
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Set the padding token
tokenizer.pad_token = tokenizer.eos_token
```

#### Freezing Model Layers

Freeze specific layers to reduce computational load:

```python
for name, param in model.named_parameters():
    if "transformer.h." in name:  # Freeze transformer blocks
        param.requires_grad = False
    elif "transformer.lm_head." in name:  # Keep the language modeling head trainable
        param.requires_grad = True
```

#### Tokenizing the Datasets

```python
tokenized_train = dataset_train.map(preprocess_function, batched=True)
tokenized_validation = dataset_validation.map(preprocess_function, batched=True)
```

#### Defining Evaluation Metrics

Use ROUGE scores for evaluation:

```python
import numpy as np
import evaluate

rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: round(v, 4) for k, v in result.items()}
    return result
```

#### Training the Model

Set up training arguments and initiate training:

```python
from transformers import TrainingArguments
from trl import SFTTrainer
from transformers import DataCollatorWithPadding

training_args = TrainingArguments(
    output_dir="gpt2_finetuned",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=6,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    report_to="none"
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_validation,
    tokenizer=tokenizer,
    args=training_args,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
```

---

## Soft Prompt Tuning (`prompt.py`)

#### Overview

Soft Prompt Tuning is a technique where we **freeze the pre-trained model parameters** and introduce a set of **trainable embeddings called soft prompts**. These soft prompts are learned during training and act as an additional context that guides the model in generating summaries without altering its original weights.

#### Implementation Details

##### 1. Defining the Soft Prompt Model

We create a custom model class `GPT2WithSoftPrompt` that wraps the pre-trained GPT-2 model and adds the soft prompt embeddings.

```python
import torch
import torch.nn as nn

class GPT2WithSoftPrompt(nn.Module):
    def __init__(self, base_model, num_soft_tokens):
        super().__init__()
        self.base_model = base_model
        self.num_soft_tokens = num_soft_tokens
        self.soft_prompt_embeddings = nn.Parameter(
            torch.randn(num_soft_tokens, base_model.config.hidden_size)
        )
        # Freeze all parameters in the base model
        for param in self.base_model.parameters():
            param.requires_grad = False
```

- **`soft_prompt_embeddings`**: A learnable parameter of shape `(num_soft_tokens, hidden_size)` that represents the embeddings of the soft prompts.
- **Freezing Base Model Parameters**: We set `requires_grad` to `False` for all parameters in the base model to prevent them from being updated during training.

##### 2. Forward Method

The `forward` method modifies the input embeddings by prepending the soft prompt embeddings and adjusts the attention masks and labels accordingly.

```python
def forward(self, input_ids=None, attention_mask=None, labels=None):
    # Get input embeddings from the base model's embedding layer
    input_embeds = self.base_model.transformer.wte(input_ids)
    batch_size = input_ids.size(0)
    
    # Expand soft prompt embeddings to match batch size
    soft_prompt_embeds = self.soft_prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
    
    # Concatenate soft prompts with input embeddings
    inputs_embeds = torch.cat([soft_prompt_embeds, input_embeds], dim=1)
    
    # Adjust attention mask to account for soft prompts
    if attention_mask is not None:
        soft_prompt_mask = torch.ones(
            batch_size, self.num_soft_tokens, dtype=attention_mask.dtype, device=attention_mask.device
        )
        attention_mask = torch.cat([soft_prompt_mask, attention_mask], dim=1)
    
    # Adjust labels to ignore soft prompt tokens
    if labels is not None:
        soft_prompt_labels = torch.full(
            (batch_size, self.num_soft_tokens), -100, dtype=labels.dtype, device=labels.device
        )
        labels = torch.cat([soft_prompt_labels, labels], dim=1)
    
    # Generate position IDs
    position_ids = torch.arange(
        inputs_embeds.size(1), dtype=torch.long, device=inputs_embeds.device
    ).unsqueeze(0).expand(batch_size, -1)
    
    # Pass modified embeddings through the base model
    outputs = self.base_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        labels=labels,
        position_ids=position_ids,
    )
    return outputs
```

- **Input Embeddings Modification**:
  - We obtain the embeddings of the input tokens.
  - The soft prompt embeddings are prepended to these input embeddings.
- **Attention Mask Adjustment**:
  - Since we've added new tokens (soft prompts), we need to adjust the attention mask to ensure the model attends to these tokens.
- **Label Adjustment**:
  - We set the labels for the soft prompt tokens to `-100` so that they are ignored in the loss computation (as per PyTorch's CrossEntropyLoss convention).
- **Position IDs**:
  - We generate new position IDs to account for the additional tokens.

##### 3. Initializing the Model

```python
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("gpt2")
num_soft_tokens = 20
model = GPT2WithSoftPrompt(base_model, num_soft_tokens=num_soft_tokens)
```

- We instantiate the base GPT-2 model.
- Set the number of soft prompt tokens (e.g., 20).
- Create an instance of our custom model with soft prompts.

##### 4. Data Preprocessing

For soft prompt tuning, we do **not** prepend any textual prompts to the input because the soft prompts serve this purpose.

```python
def preprocess_function(examples):
    inputs = examples["article"]
    model_inputs = tokenizer(
        inputs, max_length=400, truncation=True, padding='max_length'
    )
    labels = tokenizer(
        text_target=examples["highlights"], max_length=400, truncation=True, padding='max_length'
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
```

- **Inputs**: The articles are tokenized directly.
- **Labels**: The summaries are tokenized as targets.
- **No Prefix**: Unlike standard fine-tuning, we don't add any prefixes since soft prompts act as the context.

##### 5. Adjusting the Compute Metrics Function

We need to account for the soft prompt tokens when computing evaluation metrics.

```python
def compute_metrics(eval_pred):
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids

    # Discard the soft prompt tokens
    predictions = predictions[:, num_soft_tokens:, :]
    labels = labels[:, num_soft_tokens:]
    
    # Rest of the code remains the same
    predictions = np.argmax(predictions, axis=-1)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions
    ]
    result["gen_len"] = np.mean(prediction_lens)
    return {k: round(v, 4) for k, v in result.items()}
```

- **Discarding Soft Prompt Tokens**: We remove the first `num_soft_tokens` tokens from both predictions and labels before decoding.

##### 6. Training the Model

We set up the training arguments and start training using `SFTTrainer`.

```python
from transformers import TrainingArguments
from trl import SFTTrainer
from transformers import DataCollatorWithPadding

training_args = TrainingArguments(
    output_dir="gpt2_soft_prompt",
    evaluation_strategy="epoch",
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
```

- **Training Arguments**: We specify the learning rate, batch sizes, number of epochs, etc.
- **Data Collator**: Handles padding of the batches.
- **Trainer**: We use `SFTTrainer` to handle training loops, evaluation, and logging.

---

## LoRA Fine-Tuning (`lora.py`)

#### Overview

LoRA (Low-Rank Adaptation) fine-tuning introduces **low-rank decomposition matrices** into certain layers of the pre-trained model, allowing us to fine-tune the model efficiently by updating only a small number of additional parameters.

#### Implementation Details

##### 1. Setting Up LoRA Configuration

We define the LoRA configuration specifying how we want to adapt the model.

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,  # Rank of the decomposition matrices
    lora_alpha=32,  # Scaling factor
    target_modules=["c_attn"],  # Modules to apply LoRA
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

- **`r` (Rank)**: The rank of the low-rank decomposition matrices.
- **`lora_alpha`**: A scaling factor for the LoRA layers.
- **`target_modules`**: Specifies which modules in the model to apply LoRA to. For GPT-2, `"c_attn"` corresponds to the attention layers.
- **`lora_dropout`**: Dropout probability for the LoRA layers.
- **`bias`**: Whether to adapt biases (`"none"`, `"all"`, or `"lora_only"`).
- **`task_type`**: Specifies the task type, which is causal language modeling (`"CAUSAL_LM"`).
- **Printing Trainable Parameters**: We can print out the number of trainable parameters to verify that only the LoRA parameters are being updated.

##### 2. Modifying the Preprocessing Function

In LoRA fine-tuning, we need to adjust how we prepare the inputs and labels because we want the model to generate the summary based on the input article.

```python
def preprocess_function(examples):
    prefix = "summarize: "
    inputs = [prefix + doc for doc in examples["article"]]
    targets = [ex for ex in examples["highlights"]]
    max_source_length = 512 
    max_target_length = 128  

    # Tokenize inputs and targets
    tokenized_inputs = tokenizer(
        inputs, max_length=max_source_length, truncation=True, padding=False
    )
    tokenized_targets = tokenizer(
        targets, max_length=max_target_length, truncation=True, padding=False
    )

    # Combine inputs and targets
    input_ids = []
    labels = []
    for i in range(len(tokenized_inputs["input_ids"])):
        input_ids_input = tokenized_inputs["input_ids"][i]
        input_ids_target = tokenized_targets["input_ids"][i]
        # Concatenate input and target ids
        input_ids_combined = input_ids_input + input_ids_target
        input_ids_combined = input_ids_combined[:1024]  # Truncate to model's max length
        input_ids.append(input_ids_combined)

        # Create labels with -100 for the input part
        labels_combined = [-100] * len(input_ids_input) + input_ids_target
        labels_combined = labels_combined[:1024]  # Truncate labels to match inputs
        labels.append(labels_combined)

    model_inputs = {
        "input_ids": input_ids,
        "labels": labels
    }
    return model_inputs
```

- **Input and Target Tokenization**: We tokenize the inputs (articles with prefix) and targets (summaries) separately.
- **Combining Inputs and Targets**:
  - We concatenate the tokenized input IDs and target IDs to form a single sequence.
  - This allows the model to generate the target sequence conditioned on the input sequence.
- **Creating Labels**:
  - We set the labels for the input tokens to `-100` so they are ignored during loss computation.
  - The labels for the target tokens are set to the actual token IDs.

##### 3. Data Collation

We use a custom data collator to handle padding and batch preparation.

```python
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

        # Replace padding token id's in labels by -100 so they're ignored in the loss computation
        batch_labels['input_ids'][batch_labels['input_ids'] == self.tokenizer.pad_token_id] = -100

        batch = {
            'input_ids': batch_input['input_ids'],
            'attention_mask': batch_input['attention_mask'],
            'labels': batch_labels['input_ids'],
        }
        return batch
```

- **Padding**: Ensures that all sequences in the batch are of equal length by padding.
- **Labels Adjustment**: Replaces padding token IDs with `-100` in labels to ignore them during loss computation.

##### 4. Training the Model

We set up training arguments specific to LoRA fine-tuning and start the training process.

```python
from transformers import TrainingArguments
from trl import SFTTrainer

training_args = TrainingArguments(
    output_dir="gpt2_lora",
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
    eval_accumulation_steps=2,
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
```

- **Learning Rate**: LoRA fine-tuning typically requires a higher learning rate due to fewer trainable parameters.
- **Batch Sizes**: We use smaller batch sizes due to potential GPU memory constraints.
- **Evaluation Strategy**: We evaluate at the end of each epoch.
- **Load Best Model at End**: We load the best model based on evaluation loss after training.

##### 5. Evaluation Metrics

The compute metrics function remains largely the same as in standard fine-tuning, except that we adjust predictions and labels if necessary.

```python
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    # Shift predictions and labels if needed
    predictions = np.argmax(predictions, axis=-1)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    result = rouge.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    result = {k: round(v, 4) for k, v in result.items()}
    return result
```

---

## References

- [Hugging Face Transformers Documentation](https://huggingface.co/transformers/)
- [CNN/DailyMail Dataset on Hugging Face](https://huggingface.co/datasets/cnn_dailymail)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [PEFT: Parameter-Efficient Fine-Tuning](https://github.com/huggingface/peft)
- [TRL (Transformer Reinforcement Learning)](https://github.com/lvwerra/trl)

---