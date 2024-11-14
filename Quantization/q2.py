import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import time
import os

def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def compute_perplexity(model, tokenizer, dataset, max_samples=3000):
    model.eval()
    total_loss = 0.0
    total_length = 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for i, sample in enumerate(dataset):
            if i >= max_samples:
                break
            inputs = tokenizer(
                sample['text'],
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            )
            if inputs['input_ids'].size(1) == 0:
                continue
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss.item()
            total_loss += loss * inputs['input_ids'].size(1)
            total_length += inputs['input_ids'].size(1)
    if total_length == 0:
        return float('inf')
    avg_loss = total_loss / total_length
    perplexity = torch.exp(torch.tensor(avg_loss))
    return perplexity.item()

def measure_inference_latency(model, tokenizer, input_text, num_runs=10):
    device = next(model.parameters()).device
    inputs = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=512)
    if inputs['input_ids'].size(1) == 0:
        return None
    inputs = {k: v.to(device) for k, v in inputs.items()}
    total_time = 0.0
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            outputs = model(**inputs)
            total_time += time.time() - start_time
    avg_latency = total_time / num_runs
    return avg_latency

def save_model(model, model_name):
    if not os.path.exists('models'):
        os.makedirs('models')
    model.save_pretrained(f'models/{model_name}')

model_name = 'gpt2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
input_text = "This is a sample input text."
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Baseline Model (FP32)

print("Loading baseline model (FP32)...")
model_fp32 = AutoModelForCausalLM.from_pretrained(model_name)
model_fp32.eval()
model_fp32.to(device)
model_size_fp32 = get_model_size(model_fp32)
print(f"Baseline model size: {model_size_fp32:.2f} MB")
perplexity_fp32 = compute_perplexity(model_fp32, tokenizer, dataset, max_samples=30)
print(f"Baseline perplexity: {perplexity_fp32:.2f}")
latency_fp32 = measure_inference_latency(model_fp32, tokenizer, input_text)
if latency_fp32 is not None:
    print(f"Baseline inference latency: {latency_fp32*1000:.2f} ms")
else:
    print("Baseline inference latency could not be measured due to empty input.")
save_model(model_fp32, 'gpt2_fp32')

# Bitsandbytes 8-bit Quantization

print("\nApplying 8-bit quantization using bitsandbytes...")
bnb_config_8bit = BitsAndBytesConfig(load_in_8bit=True)
model_8bit = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config_8bit,
    device_map='auto'
)
model_8bit.eval()
model_size_8bit = get_model_size(model_8bit)
print(f"Model size after 8-bit quantization: {model_size_8bit:.2f} MB")
perplexity_8bit = compute_perplexity(model_8bit, tokenizer, dataset, max_samples=30)
print(f"Perplexity after 8-bit quantization: {perplexity_8bit:.2f}")
latency_8bit = measure_inference_latency(model_8bit, tokenizer, input_text)
if latency_8bit is not None:
    print(f"Inference latency after 8-bit quantization: {latency_8bit*1000:.2f} ms")
else:
    print("Inference latency after 8-bit quantization could not be measured due to empty input.")
save_model(model_8bit, 'gpt2_8bit')
print("\nChanges after 8-bit quantization:")
print(f"Perplexity change: {perplexity_8bit - perplexity_fp32:.2f}")
if latency_fp32 is not None and latency_8bit is not None:
    print(f"Inference latency change: {latency_8bit - latency_fp32:.6f} seconds")
else:
    print("Inference latency change could not be computed.")
print(f"Model size reduction: {model_size_fp32 - model_size_8bit:.2f} MB")

# Bitsandbytes 4-bit Quantization

print("\nApplying 4-bit quantization using bitsandbytes...")
bnb_config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='fp4'  
)

model_4bit = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config_4bit,
    device_map='auto'
)
model_4bit.eval()
model_size_4bit = get_model_size(model_4bit)
print(f"Model size after 4-bit quantization: {model_size_4bit:.2f} MB")
perplexity_4bit = compute_perplexity(model_4bit, tokenizer, dataset, max_samples=30)
print(f"Perplexity after 4-bit quantization: {perplexity_4bit:.2f}")
latency_4bit = measure_inference_latency(model_4bit, tokenizer, input_text)
if latency_4bit is not None:
    print(f"Inference latency after 4-bit quantization: {latency_4bit*1000:.2f} ms")
else:
    print("Inference latency after 4-bit quantization could not be measured due to empty input.")
save_model(model_4bit, 'gpt2_4bit')
print("\nChanges after 4-bit quantization:")
print(f"Perplexity change: {perplexity_4bit - perplexity_fp32:.2f}")
if latency_fp32 is not None and latency_4bit is not None:
    print(f"Inference latency change: {latency_4bit - latency_fp32:.6f} seconds")
else:
    print("Inference latency change could not be computed.")
print(f"Model size reduction: {model_size_fp32 - model_size_4bit:.2f} MB")

# NF4 Quantization

print("\nApplying NF4 quantization using bitsandbytes...")
bnb_config_nf4 = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4'  
)
model_nf4 = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config_nf4,
    device_map='auto'
)
model_nf4.eval()
model_size_nf4 = get_model_size(model_nf4)
print(f"Model size after NF4 quantization: {model_size_nf4:.2f} MB")
perplexity_nf4 = compute_perplexity(model_nf4, tokenizer, dataset, max_samples=30)
print(f"Perplexity after NF4 quantization: {perplexity_nf4:.2f}")
latency_nf4 = measure_inference_latency(model_nf4, tokenizer, input_text)
if latency_nf4 is not None:
    print(f"Inference latency after NF4 quantization: {latency_nf4*1000:.2f} ms")
else:
    print("Inference latency after NF4 quantization could not be measured due to empty input.")
save_model(model_nf4, 'gpt2_nf4')
print("\nChanges after NF4 quantization:")
print(f"Perplexity change: {perplexity_nf4 - perplexity_fp32:.2f}")
if latency_fp32 is not None and latency_nf4 is not None:
    print(f"Inference latency change: {latency_nf4 - latency_fp32:.6f} seconds")
else:
    print("Inference latency change could not be computed.")
print(f"Model size reduction: {model_size_fp32 - model_size_nf4:.2f} MB")

print("\n--- Analysis ---")
print(f"Baseline model size: {model_size_fp32:.2f} MB")
print(f"8-bit model size: {model_size_8bit:.2f} MB")
print(f"4-bit model size: {model_size_4bit:.2f} MB")
print(f"NF4 model size: {model_size_nf4:.2f} MB")
print("\nPerplexity Comparison:")
print(f"Baseline perplexity: {perplexity_fp32:.2f}")
print(f"8-bit perplexity: {perplexity_8bit:.2f}")
print(f"4-bit perplexity: {perplexity_4bit:.2f}")
print(f"NF4 perplexity: {perplexity_nf4:.2f}")
print("\nInference Latency Comparison:")
if latency_fp32 is not None:
    print(f"Baseline latency: {latency_fp32*1000:.2f} ms")
if latency_8bit is not None:
    print(f"8-bit latency: {latency_8bit*1000:.2f} ms")
if latency_4bit is not None:
    print(f"4-bit latency: {latency_4bit*1000:.2f} ms")
if latency_nf4 is not None:
    print(f"NF4 latency: {latency_nf4*1000:.2f} ms")
