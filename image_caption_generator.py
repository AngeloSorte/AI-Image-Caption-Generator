# 📦 Install required libraries
!pip install transformers pillow

# 📚 Import libraries
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests

# ⚙️ Load the pre-trained model and processor from Hugging Face
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# 🖼️ Load an image from a URL
image_url = "https://images.unsplash.com/photo-1547721064-da6cfb341d50"
image = Image.open(requests.get(image_url, stream=True).raw)

# ✏️ Process the image and generate a caption
inputs = processor(image, return_tensors="pt")
out = model.generate(**inputs)

# 📢 Decode the output and print the caption
caption = processor.decode(out[0], skip_special_tokens=True)
print(f"🖼️ Caption: {caption}")
