import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import time

def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def compute_perplexity(model, tokenizer, dataset, max_samples=3000):
    model.eval()
    total_loss = 0.0
    total_length = 0
    with torch.no_grad():
        for i, sample in enumerate(dataset):
            if i >= max_samples:
                break
            inputs = tokenizer(sample['text'], return_tensors='pt', truncation=True, max_length=512)
            if inputs['input_ids'].size(1) == 0:
                continue
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
    inputs = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=512)
    if inputs['input_ids'].size(1) == 0:
        return None
    total_time = 0.0
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            outputs = model(**inputs)
            total_time += time.time() - start_time
    avg_latency = total_time / num_runs
    return avg_latency

def quantize_model_weights(model):
    quantized_state_dict = {}
    scales = {}
    zero_points = {}
    for name, param in model.state_dict().items():
        if param.dtype == torch.float32 or param.dtype == torch.float16:
            min_val = param.min()
            max_val = param.max()
            qmin = -128
            qmax = 127
            scale = (max_val - min_val) / (qmax - qmin)
            if scale == 0:
                scale = 1e-8  
            zero_point = qmin - min_val / scale
            zero_point = int(zero_point.round())
            q_param = ((param / scale) + zero_point).round().clamp(qmin, qmax).to(torch.int8)
            quantized_state_dict[name] = q_param
            scales[name] = scale
            zero_points[name] = zero_point
        else:
            quantized_state_dict[name] = param
    return quantized_state_dict, scales, zero_points

def dequantize_model_weights(quantized_state_dict, scales, zero_points):
    state_dict = {}
    for name, param in quantized_state_dict.items():
        if param.dtype == torch.int8:
            scale = scales[name]
            zero_point = zero_points[name]
            dequantized_param = (param.to(torch.float32) - zero_point) * scale
            state_dict[name] = dequantized_param
        else:
            state_dict[name] = param
    return state_dict

def quantize_selective_mlp(model):
    quantized_state_dict = {}
    scales = {}
    zero_points = {}
    for name, param in model.state_dict().items():
        if 'mlp' in name and (param.dtype == torch.float32 or param.dtype == torch.float16):
            min_val = param.min()
            max_val = param.max()
            qmin = -128
            qmax = 127
            scale = (max_val - min_val) / (qmax - qmin)
            if scale == 0:
                scale = 1e-8  
            zero_point = qmin - min_val / scale
            zero_point = int(zero_point.round())
            q_param = ((param / scale) + zero_point).round().clamp(qmin, qmax).to(torch.int8)
            quantized_state_dict[name] = q_param
            scales[name] = scale
            zero_points[name] = zero_point
        else:
            quantized_state_dict[name] = param
    return quantized_state_dict, scales, zero_points

def quantize_selective_decoder(model, layers_to_quantize):
    quantized_state_dict = {}
    scales = {}
    zero_points = {}
    for name, param in model.state_dict().items():
        quantize_param = False
        for layer_idx in layers_to_quantize:
            layer_name = f'transformer.h.{layer_idx}.'
            if name.startswith(layer_name):
                quantize_param = True
                break
        if quantize_param and (param.dtype == torch.float32 or param.dtype == torch.float16):
            min_val = param.min()
            max_val = param.max()
            qmin = -128
            qmax = 127
            scale = (max_val - min_val) / (qmax - qmin)
            if scale == 0:
                scale = 1e-8  
            zero_point = qmin - min_val / scale
            zero_point = int(zero_point.round())
            q_param = ((param / scale) + zero_point).round().clamp(qmin, qmax).to(torch.int8)
            quantized_state_dict[name] = q_param
            scales[name] = scale
            zero_points[name] = zero_point
        else:
            quantized_state_dict[name] = param
    return quantized_state_dict, scales, zero_points

model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

model_size_before = get_model_size(model)
print(f"Model size before quantization: {model_size_before:.2f} MB")

dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

perplexity_before = compute_perplexity(model, tokenizer, dataset, max_samples=30)
print(f"Perplexity before quantization: {perplexity_before:.2f}")

latency_before = measure_inference_latency(model, tokenizer, "This is a sample input text.")
if latency_before is not None:
    print(f"Inference latency before quantization: {latency_before*1000:.2f} ms")
else:
    print("Inference latency before quantization could not be measured due to empty input.")

# Whole-Model Quantization
print("\nPerforming whole-model quantization...")
quantized_state_dict, scales, zero_points = quantize_model_weights(model)
torch.save({'state_dict': quantized_state_dict, 'scales': scales, 'zero_points': zero_points}, 'gpt2_whole_quantized.pth')
quantized_param_size = sum(param.nelement() * param.element_size() for param in quantized_state_dict.values())
model_size_after = quantized_param_size / 1024**2
print(f"Model size after whole-model quantization: {model_size_after:.2f} MB")
dequantized_state_dict = dequantize_model_weights(quantized_state_dict, scales, zero_points)
model.load_state_dict(dequantized_state_dict)
perplexity_after = compute_perplexity(model, tokenizer, dataset, max_samples=3000)
print(f"Perplexity after whole-model quantization: {perplexity_after:.2f}")
latency_after = measure_inference_latency(model, tokenizer, "This is a sample input text.")
if latency_after is not None:
    print(f"Inference latency after whole-model quantization: {latency_after*1000:.2f} ms")
else:
    print("Inference latency after quantization could not be measured due to empty input.")
print("\nChanges after whole-model quantization:")
print(f"Perplexity increase: {perplexity_after - perplexity_before:.2f}")
if latency_before is not None and latency_after is not None:
    print(f"Inference latency change: {latency_after - latency_before:.6f} seconds")
else:
    print("Inference latency change could not be computed.")
print(f"Model size reduction: {model_size_before - model_size_after:.2f} MB")

# Selective Quantization - MLP Layers
model = GPT2LMHeadModel.from_pretrained(model_name)
print("\nPerforming selective quantization on MLP layers...")
quantized_state_dict_mlp, scales_mlp, zero_points_mlp = quantize_selective_mlp(model)
torch.save({'state_dict': quantized_state_dict_mlp, 'scales': scales_mlp, 'zero_points': zero_points_mlp}, 'gpt2_mlp_quantized.pth')
quantized_param_size_mlp = sum(param.nelement() * param.element_size() for param in quantized_state_dict_mlp.values())
model_size_after_mlp = quantized_param_size_mlp / 1024**2
print(f"Model size after MLP quantization: {model_size_after_mlp:.2f} MB")
dequantized_state_dict_mlp = dequantize_model_weights(quantized_state_dict_mlp, scales_mlp, zero_points_mlp)
model.load_state_dict(dequantized_state_dict_mlp)
perplexity_after_mlp = compute_perplexity(model, tokenizer, dataset, max_samples=3000)
print(f"Perplexity after MLP quantization: {perplexity_after_mlp:.2f}")
latency_after_mlp = measure_inference_latency(model, tokenizer, "This is a sample input text.")
if latency_after_mlp is not None:
    print(f"Inference latency after MLP quantization: {latency_after_mlp*1000:.2f} ms")
else:
    print("Inference latency after MLP quantization could not be measured due to empty input.")
print("\nChanges after MLP quantization:")
print(f"Perplexity increase: {perplexity_after_mlp - perplexity_before:.2f}")
if latency_before is not None and latency_after_mlp is not None:
    print(f"Inference latency change: {latency_after_mlp - latency_before:.6f} seconds")
else:
    print("Inference latency change could not be computed.")
print(f"Model size reduction: {model_size_before - model_size_after_mlp:.2f} MB")

# Selective Quantization - Decoder Layers
model = GPT2LMHeadModel.from_pretrained(model_name)
decoder_layers = list(range(12))  # 12 layers
print("\nPerforming selective quantization on decoder layers...")
quantized_state_dict_decoder, scales_decoder, zero_points_decoder = quantize_selective_decoder(model, decoder_layers)
torch.save({'state_dict': quantized_state_dict_decoder, 'scales': scales_decoder, 'zero_points': zero_points_decoder}, 'gpt2_decoder_quantized.pth')
quantized_param_size_decoder = sum(param.nelement() * param.element_size() for param in quantized_state_dict_decoder.values())
model_size_after_decoder = quantized_param_size_decoder / 1024**2
print(f"Model size after decoder quantization: {model_size_after_decoder:.2f} MB")
dequantized_state_dict_decoder = dequantize_model_weights(quantized_state_dict_decoder, scales_decoder, zero_points_decoder)
model.load_state_dict(dequantized_state_dict_decoder)
perplexity_after_decoder = compute_perplexity(model, tokenizer, dataset, max_samples=3000)
print(f"Perplexity after decoder quantization: {perplexity_after_decoder:.2f}")
latency_after_decoder = measure_inference_latency(model, tokenizer, "This is a sample input text.")
if latency_after_decoder is not None:
    print(f"Inference latency after decoder quantization: {latency_after_decoder*1000:.2f} ms")
else:
    print("Inference latency after decoder quantization could not be measured due to empty input.")
print("\nChanges after decoder quantization:")
print(f"Perplexity increase: {perplexity_after_decoder - perplexity_before:.2f}")
if latency_before is not None and latency_after_decoder is not None:
    print(f"Inference latency change: {latency_after_decoder - latency_before:.6f} seconds")
else:
    print("Inference latency change could not be computed.")
print(f"Model size reduction: {model_size_before - model_size_after_decoder:.2f} MB")