from datasets import load_dataset
from PIL import Image
from transformers import BlipProcessor
from transformers import BlipForConditionalGeneration
from transformers import TrainingArguments, Trainer
from huggingface_hub import notebook_login

data_files = {
    "train": "BLIP/train/captions.csv",
    "eval": "BLIP/eval/captions.csv"
}

# Specify delimiter for TSV files
dataset = load_dataset("csv", data_files=data_files, delimiter="\t")


print(dataset["train"])
from transformers import BlipProcessor

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")


def preprocess(example, split="train"):
    # Load the image
    image_path = f"BLIP/{split}/images/{example['filename']}"
    image = Image.open(image_path).convert("RGB")

    # Process image and caption
    encoding = processor(images=image, text=example['caption'], return_tensors="pt", padding=True, truncation=True)
    encoding["labels"] = encoding["input_ids"].clone()
    return {
        "pixel_values": encoding["pixel_values"].squeeze(),
        "input_ids": encoding["input_ids"].squeeze(),
        "labels": encoding["labels"].squeeze()
    }


# Apply preprocessing to both train and eval splits
dataset["train"] = dataset["train"].map(lambda x: preprocess(x, "train"), remove_columns=["filename", "caption"])
dataset["eval"] = dataset["eval"].map(lambda x: preprocess(x, "eval"), remove_columns=["filename", "caption"])

print(dataset)

model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
training_args = TrainingArguments(
    output_dir="./blip-finetuned",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["eval"]  # if you have a validation split
)
trainer.train()
notebook_login()
model.push_to_hub("Quadrant/qhub-blip-finetuned-model")
processor.push_to_hub("Quadrant/qhub-blip-finetuned-processor")
