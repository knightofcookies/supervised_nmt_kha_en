from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    MarianMTModel,
    MarianConfig,
)
from transformers import TrainingArguments, Trainer
import torch
from datasets import load_dataset

# Specify the directory where you saved the model
SAVE_DIRECTORY = "."

# Load the model and tokenizer from the local directory
model = AutoModelForCausalLM.from_pretrained(SAVE_DIRECTORY)
tokenizer = AutoTokenizer.from_pretrained(SAVE_DIRECTORY)

# Add padding token to the tokenizer
tokenizer.add_special_tokens({"pad_token": "[PAD]"})

# Resize the model embeddings
model.resize_token_embeddings(len(tokenizer))

# Load the parallel corpus
data = load_dataset("text", data_files={"train": "../../datasets/pc_15k.tsv"})


# Preprocess the data
def preprocess_function(examples):
    source, target = zip(*[line.split("\t\t\t\t\t") for line in examples["text"]])
    model_inputs = tokenizer(
        source, max_length=128, truncation=True, padding="max_length"
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            target, max_length=128, truncation=True, padding="max_length"
        )
    model_inputs["labels"] = labels["input_ids"]

    # Convert to tensors
    for key in model_inputs:
        model_inputs[key] = torch.tensor(model_inputs[key])

    return model_inputs


data = data.map(
    preprocess_function, batched=True, remove_columns=data["train"].column_names
)

# Update the MarianConfig with the new vocabulary size
en_kha_config = MarianConfig(
    vocab_size=len(tokenizer),
    decoder_start_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    max_length=128,
    early_stopping=True,
    num_beams=4,
    bos_token_id=tokenizer.bos_token_id,
)

kha_en_config = MarianConfig(
    vocab_size=len(tokenizer),
    decoder_start_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    max_length=128,
    early_stopping=True,
    num_beams=4,
    bos_token_id=tokenizer.bos_token_id,
)

# Create two MarianMTModel instances for each translation direction
en_kha_model = MarianMTModel(en_kha_config)
kha_en_model = MarianMTModel(kha_en_config)

# Move the model to the GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
en_kha_model.to(DEVICE)
kha_en_model.to(DEVICE)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./en_kha_kha_en_model",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    save_steps=1000,
    logging_dir="./logs",
    learning_rate=2e-5,
)


# Define a custom Trainer class to handle two models
class MultiModelTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.en_kha_model = en_kha_model
        self.kha_en_model = kha_en_model

    def compute_loss(self, model, inputs, return_outputs=False):
        # Inputs should already be tensors, so no need for conversion

        # Compute loss for English-Khasi translation
        en_kha_outputs = self.en_kha_model(**inputs)
        en_kha_loss = en_kha_outputs.loss

        # Compute loss for Khasi-English translation (reverse the inputs)
        reversed_inputs = {k: v for k, v in inputs.items()}
        reversed_inputs["input_ids"], reversed_inputs["labels"] = (
            reversed_inputs["labels"],
            reversed_inputs["input_ids"],
        )
        kha_en_outputs = self.kha_en_model(**reversed_inputs)
        kha_en_loss = kha_en_outputs.loss

        # Combine the losses
        loss = (en_kha_loss + kha_en_loss) / 2

        return (loss, (en_kha_outputs, kha_en_outputs)) if return_outputs else loss


# Create the Trainer
trainer = MultiModelTrainer(
    model=model,  # This is a dummy model, not used for training
    args=training_args,
    train_dataset=data["train"],
    # Remove the data_collator as it's no longer needed
)


# Train the models
trainer.train()

# Save the trained models
en_kha_model.save_pretrained("./en_kha_model")
kha_en_model.save_pretrained("./kha_en_model")

print("Training completed. Models saved to ./en_kha_model and ./kha_en_model")
