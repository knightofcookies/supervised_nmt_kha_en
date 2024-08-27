from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline, PegasusXConfig
import torch

# Specify the directory where you saved the model
save_directory = "."

# Load the model and tokenizer from the local directory
config = PegasusXConfig.from_pretrained(save_directory)
model = AutoModelForSeq2SeqLM.from_pretrained(save_directory, config=config)
tokenizer = AutoTokenizer.from_pretrained(save_directory)

print(torch.cuda.is_available())

# Move the model to the GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Create a summarization pipeline
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

# Start the summarization loop
while True:
    # Get user input
    user_input = input("Enter text to summarize: ")

    # Generate summary
    summary = summarizer(user_input, max_length=100, min_length=30, do_sample=False)[0]['summary_text']

    # Print the summary
    print("Summary: ", summary)