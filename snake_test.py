from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import torch
import torchvision.transforms as T
import torch.nn.functional as F

# Load the model and processor
model = AutoModelForImageClassification.from_pretrained("results_n17").to("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained("results_n17", use_fast=True)
labels = list(model.config.id2label.values())

# Load and preprocess the image
image_path = "./ModelTesterImage/Monocled_Cobra_Tester.jpg"
print(f"Input image: {image_path}")
image = Image.open(image_path).convert("RGB")

transform = T.Compose([
    T.Resize((processor.size["height"], processor.size["width"])),
    T.ToTensor(),
    T.Normalize(mean=processor.image_mean, std=processor.image_std)
])
input_tensor = transform(image).unsqueeze(0).to(model.device)

# Predict
with torch.no_grad():
    outputs = model(pixel_values=input_tensor)
    logits = outputs.logits
    probs = F.softmax(logits, dim=-1).squeeze()

# Find predicted class
predicted_idx = torch.argmax(probs).item()
predicted_label = labels[predicted_idx]
predicted_confidence = probs[predicted_idx].item() * 100

# Print all class probabilities
print("All class probabilities:")
sorted_probs, indices = torch.sort(probs, descending=True)
for idx in indices:
    label = labels[idx]
    percentage = probs[idx].item() * 100
    print(f"{label}: {percentage:.1f}%")

# Print main result
print(f"\nPredicted class: {predicted_label} ({predicted_confidence:.1f}%)\n")