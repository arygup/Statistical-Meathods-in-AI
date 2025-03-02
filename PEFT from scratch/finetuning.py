from datasets import load_dataset
import random 
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding, TrainingArguments
import numpy as np
import evaluate
from trl import SFTTrainer

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

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

for name, param in model.named_parameters():
    if "transformer.h." in name:  
        param.requires_grad = False  
    elif "transformer.lm_head." in name:  
        param.requires_grad = True  

prefix = "summarize: "

def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=400, truncation=True, padding='max_length')
    labels = tokenizer(text_target=examples["highlights"], max_length=400, truncation=True, padding='max_length')
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = dataset_train.map(preprocess_function, batched=True)
tokenized_validation = dataset_validation.map(preprocess_function, batched=True)
tokenized_test = dataset_test.map(preprocess_function, batched=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    return {k: round(v, 4) for k, v in result.items()}

training_args = TrainingArguments(
    output_dir="gpt2_test",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=6,
    per_device_eval_batch_size=1,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=20,
    report_to="none",
    fp16=True,
    eval_accumulation_steps=8
)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id
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