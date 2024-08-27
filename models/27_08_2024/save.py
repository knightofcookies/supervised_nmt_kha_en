from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer (if not already loaded)
model_name = "EleutherAI/pythia-1.4b"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Specify the directory where you want to save the model
save_directory = "."

# Save the model and tokenizer
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)