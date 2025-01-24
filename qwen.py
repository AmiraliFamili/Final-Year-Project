import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)

# Load the model with CPU (avoiding the `device_map` parameter)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    offload_folder="offload_dir"
).to("cpu")

# Example input
input_text = "translate what you just asked into english pls"

# Tokenize the input
input_ids = tokenizer.encode(input_text, return_tensors="pt").to("cpu")

# Generate a response
output_ids = model.generate(input_ids, max_length=50, num_return_sequences=1)

# Decode the output
response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print(response)
