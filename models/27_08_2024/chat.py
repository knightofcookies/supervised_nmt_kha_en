from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Specify the directory where you saved the model
save_directory = "."

# Load the model and tokenizer from the local directory
model = AutoModelForCausalLM.from_pretrained(save_directory)
tokenizer = AutoTokenizer.from_pretrained(save_directory)

print(torch.cuda.is_available())

# Move the model to the GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Create a text generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Start the chat loop
while True:
    # Get user input
    user_input = input("User: ")

    # Generate response
    response = generator(user_input, max_length=100, temperature=0.7, num_return_sequences=1)[0]['generated_text']

    # Print the response
    print("Machine: ", response)