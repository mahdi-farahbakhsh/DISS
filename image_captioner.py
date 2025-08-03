from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch
import os

# Load model and processor
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")

image_path = "./imagenet_test_data/images/"

# Create a directory to save the captions
os.makedirs("captions", exist_ok=True)

n_images = 5000

for i, image_file in enumerate(sorted(os.listdir(image_path))):  # change to sorted list

    if i >= n_images:
        break

    image = Image.open(os.path.join(image_path, image_file)).convert('RGB')

    # Prepare input and generate caption
    text_prompt = "a detailed photograph of"
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(model.device)

    out = model.generate(**inputs, max_new_tokens=50)

    print("out: ", out)

    caption = processor.decode(out[0], skip_special_tokens=True)

    print("Caption:", caption)

    # Save caption to a file in append mode

    image_file_name = image_file.replace('.jpg', '.txt')

    with open(f"./imagenet_test_data/captions/{image_file_name}", "w") as f:
        f.write(f"{caption}")


