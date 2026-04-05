from huggingface_hub import HfApi, create_repo, upload_folder
import os

# ================= CONFIG =================
HF_USERNAME = "teohyc"   # change this
REPO_NAME = "Pneumonia_Lung_Disease_Detection-ViT"   

TEMP_DIR = "hf_upload_vit"  

#prepare folder
os.makedirs(TEMP_DIR, exist_ok=True)

#inference and model code
import shutil

#model file
shutil.copy("ViT_model.py", os.path.join(TEMP_DIR, "ViT_model.py"))

#model weights 
shutil.copy("vit_with_generated.pth", os.path.join(TEMP_DIR, "ViT_model_weights.pth"))



#readme
readme = """
# Vision Transformer (ViT) for Lung Pneumonia Detection

This is a Vision Transformer (ViT) model designed for detecting pneumonia in chest X-rays. The model takes a chest X-ray image as input, processes it through the ViT architecture, and outputs the predicted class probabilities.
Classes include:
- Normal
- Bacterial Pneumonia
- Viral Pneumonia
- COVID-19

Training data from https://data.mendeley.com/datasets/9xkhgts2s6/4
Full project file at https://github.com/teohyc/pneumonia_disease_agent


## Usage 

```python
from PIL import Image
import torch
from torchvision import transforms
from ViT_model import ViT  # replace with the actual class name from ViT_model.py

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit_model = ViT()
vit_model.load_state_dict(torch.load("vit_with_generated.pth", map_location=device))
vit_model = vit_model.to(device)
vit_model.eval()

def vit_inference(image: Image.Image) -> dict:
    

    class_names = ["Normal", "Bacterial Pneumonia", "Viral Pneumonia", "COVID-19"]

    
    # load and preprocess image
    image = image.convert("L")  # Convert to grayscale
        
    # define preprocessing transform
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
        
        
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    image_tensor = image_tensor.to(device)
        
    #inference
    with torch.no_grad():
        outputs, attn_weights_all = vit_model(image_tensor)
        print('vit outputs:', outputs)
        if torch.isnan(outputs).any(): #debugging
            print('Warning: NaNs in outputs')

        probabilities = torch.softmax(outputs, dim=1)
        print('probabilities:', probabilities)
            
    #probabilities for all classes and transfer to cpu and to numpy array
    probs = probabilities[0].cpu().numpy()
    print('probs numpy:', probs)
        
    #list of class_name, probability and sort by probability descending
    class_probs = [(class_names[i], float(probs[i])) for i in range(len(class_names))]
    class_probs.sort(key=lambda x: x[1], reverse=True)
        
    #top predicted class and confidence
    predicted_class = class_probs[0][0]
    confidence_score = class_probs[0][1]
    print(predicted_class, confidence_score)


image_path = "t_viral_test.jpg"  # replace with your test image path
image = Image.open(image_path)
vit_inference(image)
```"""

with open(os.path.join(TEMP_DIR, "README.md"), "w") as f:
    f.write(readme)

#create repo and upload

api = HfApi()

repo_id = f"{HF_USERNAME}/{REPO_NAME}"

create_repo(repo_id, exist_ok=True)

print(f"Uploading to {repo_id}...")

upload_folder(
folder_path=TEMP_DIR,
repo_id=repo_id,
repo_type="model"
)

print("\nUpload complete!")
print(f"https://huggingface.co/{repo_id}")