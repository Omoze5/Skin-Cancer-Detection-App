import streamlit as st
import torch
import torch.nn as nn
import librosa
import numpy as np
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms



st.markdown("<h2 style='text-align: center; color: blue;'>Melanoma Skin Cancer Detection App</h2>", unsafe_allow_html=True)



# Define the model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3),           
            nn.ReLU(),                    
            nn.Conv2d(32, 64, 3),          
            nn.ReLU(),                     
            nn.Conv2d(64, 128, 3),         
            nn.ReLU(),                     
            nn.Flatten(),                  
            nn.Linear(128 * 122 * 122, 10),
            nn.ReLU(),                     
            nn.Linear(10, 2)               
        )
    
    def forward(self, x):
        return self.model(x)


import gdown

url = "https://drive.google.com/uc?id=1BH2Ro3CSL2zVdtN6YyBwwMfh9AT4KhqY"
output = "model.pth"
gdown.download(url, output, quiet=False)


# Load model
model = MyModel()  
model.load_state_dict(torch.load(output))
model.eval()


# Define labels
labels = ["Benign", "Malignant"]


# Define preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Streamlit UI
st.write("Upload an image to predict if it is benign or malignant.")

uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")

    # Preprocess the image
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        prediction = output.argmax().item()

    # Display the prediction
    st.write(f"Predicted Class: {labels[prediction]}")
