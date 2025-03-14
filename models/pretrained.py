from transformers import T5Tokenizer, T5ForConditionalGeneration

# Define the model name
model_name = "t5-base"

# Load the pre-trained model and tokenizer
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Save the pre-trained model and tokenizer to the 'pretrained' directory
model.save_pretrained("models/pretrained/t5-base")
tokenizer.save_pretrained("models/pretrained/t5-base")

print("Pre-trained model and tokenizer saved to 'models/pretrained/t5-base'")