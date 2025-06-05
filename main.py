import torch
import numpy as np
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
from torchvision.transforms import Compose, RandomResizedCrop, ToTensor, Normalize
import evaluate
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
dataset_dict = load_dataset("imagefolder", data_dir="snake_images")
full_dataset = dataset_dict["train"]

# Split into train (80%) and temp (20%)
split_1 = full_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split_1["train"]
temp_dataset = split_1["test"]

# Split temp into val (15%) and test (5%)
split_2 = temp_dataset.train_test_split(test_size=0.25, seed=42)
val_dataset = split_2["train"]
test_dataset = split_2["test"]

# Load ViT image processor
checkpoint = "google/vit-base-patch16-224-in21k"
image_processor = AutoImageProcessor.from_pretrained(checkpoint, use_fast=True)

# Define torchvision transforms
normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
size = image_processor.size.get("shortest_edge") or (image_processor.size["height"], image_processor.size["width"])
torch_transforms = Compose([
    RandomResizedCrop(size),
    ToTensor(),
    normalize
])

# Batched transform function
def apply_transforms(examples):
    examples["pixel_values"] = [torch_transforms(img.convert("RGB")) for img in examples["image"]]
    del examples["image"]
    return examples

# Apply transforms to all splits
for name in ["train_dataset", "val_dataset", "test_dataset"]:
    dataset = locals()[name]
    dataset.set_format(type="python")
    dataset = dataset.with_transform(apply_transforms)
    locals()[name] = dataset

# Label mappings
labels = train_dataset.features["label"].names
label2id = {label: str(i) for i, label in enumerate(labels)}
id2label = {str(i): label for i, label in enumerate(labels)}

# Load ViT model
model = AutoModelForImageClassification.from_pretrained(
    checkpoint,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
).to(device)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    save_strategy="epoch",
    num_train_epochs=5,
    learning_rate=3e-5,
    fp16=False,  # Set to False for CPU
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    report_to="none",
    remove_unused_columns=False,
)

# Accuracy metric
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=preds, references=labels)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate on test set
results = trainer.evaluate(test_dataset)
print("Test results:", results)

# Predict on test set
preds_output = trainer.predict(test_dataset)
y_pred = np.argmax(preds_output.predictions, axis=1)
y_true = preds_output.label_ids

# Plot confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/confusion_matrix.png")
plt.close()
print("Confusion matrix saved to: results/confusion_matrix.png")

# Save model and processor for inference
model.save_pretrained("results")
image_processor.save_pretrained("results")
print("Model and processor saved to 'results/' for future use.")
