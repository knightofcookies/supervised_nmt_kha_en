from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Specify the directory where you saved the model
save_directory = "."  # Assuming you saved it in the current directory

# Load the model and tokenizer from the local directory
model = AutoModelForCausalLM.from_pretrained(save_directory)
tokenizer = AutoTokenizer.from_pretrained(save_directory)

# Move the model to the GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Print the model architecture
print(model)