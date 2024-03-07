import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import plotly.express as px
import time
from keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image

dim_z = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ngpu = 1

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        
        self.structure = nn.Sequential(
            nn.ConvTranspose2d(in_channels=dim_z, out_channels=512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )
        
    def forward(self, z):
        return self.structure(z)
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu  
        self.structure = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),  

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), 
            nn.LeakyReLU(0.2, inplace=True),

    
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),


            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),


            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid() 
        )

    def forward(self, img):
        return self.structure(img)

model_G = Generator(ngpu=1)
model_G.to(device)

model_D = Discriminator(ngpu=1)
model_D.to(device)
# Define paths to pre-trained models
path_train = "./Model/celeba_gen_train.pth"
path_retrain = "./Model/generator_retrained.pth"
path_model_train = "./Model/celeba_gen_1.pth"
from test import Generator
from test import Discriminator
# Load the saved GAN model
gan = load_model('./Model/gan_model_hyper.h5')
generator = gan.layers[1]

def load_and_evaluate_model(path):
    model_instance = Generator(ngpu=1)
    model_instance.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model_instance = model_instance.eval()
    return model_instance

model_train = load_and_evaluate_model(path_train)
model_retrain = load_and_evaluate_model(path_retrain)
model_model_train = load_and_evaluate_model(path_model_train)
from keras.models import load_model
from PIL import ImageEnhance
def resize_image(image, basewidth=200):
    # Chuyển đổi ảnh từ dạng numpy array sang PIL Image
    img = Image.fromarray((image * 255).astype('uint8'))

    # Tính toán kích thước mới
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))

    # Thay đổi kích thước ảnh
    img = img.resize((basewidth, hsize), Image.BICUBIC)

    # Chuyển đổi ảnh từ dạng PIL Image sang numpy array
    img = np.array(img) / 255.0

    return img

def app():
    selected_model = None
    train_model = st.sidebar.selectbox("Select Model", ["GAN", "DCGAN"])
    if train_model == "GAN":
        st.header("GAN Image Generator")
        start_time = time.time()
        # Number of images you want to generate
        num_images = 14
        # Generate images using the generator
        generated_images = []

        for _ in range(num_images):
            latent_vector = np.random.normal(size=(1, 32))
            generated_image = generator.predict(latent_vector)
            generated_images.append(generated_image)

        # Display generated images in a row without captions
        st.image(
            [Image.fromarray((img[0] * 255).astype('uint8')).resize((100, 100)) for img in generated_images],
            width=100,  # Set the desired width
        )
        end_time = time.time()
        execution_time = (end_time - start_time)*1000
        st.write(f"Thời gian chạy: {execution_time:.2f} mili giây")
    elif train_model == "DCGAN":
        st.header("DCGAN Image Generator")
        train_option = st.sidebar.selectbox("Select Option", ["Model", "Train Model", "Remodel"])
        if train_option == "Model":
            start_time = time.time()
            selected_model = model_model_train
            end_time = time.time()
            execution_time = (end_time - start_time)*1000
            st.write(f"Thời gian chạy: {execution_time:.2f} mili giây")
        elif train_option == "Remodel":
            start_time = time.time()
            selected_model = model_retrain
            end_time = time.time()
            execution_time = (end_time - start_time)*1000
            st.write(f"Thời gian chạy: {execution_time:.2f} mili giây")
        elif train_option == "Train Model":
            start_time = time.time()
            selected_model = model_train
            end_time = time.time()
            execution_time = (end_time - start_time)*1000
            st.write(f"Thời gian chạy: {execution_time:.2f} mili giây")
    if selected_model is not None:  # Check if selected_model is defined
        # Number of samples to generate for each selection
        num_samples = 14
        # Allow users to control input vectors interactively
        st.sidebar.header("Customize Noise Vector")
        noise_vector = torch.randn(num_samples, dim_z, 1, 1, device=device)
        noise_vector_widget = st.sidebar.slider("Adjust Noise Vector", -3.0, 3.0, 0.0)
        noise_vector += noise_vector_widget
        generated_images_list = []
        num_image = 1
        for i in range(num_image):
            with torch.no_grad():
                generated_images = selected_model(noise_vector)
            # Normalize the pixel values to [0, 1]
            generated_images = (generated_images + 1) / 2

            generated_images_list.append(generated_images.permute(0, 2, 3, 1).cpu().numpy())

        # Combine the images into a single numpy array with shape (H, W, C)
        combined_images = np.concatenate(generated_images_list, axis=0)
        image_width = 100
        # Display the combined row of images
        st.image(combined_images, use_column_width=False,width=image_width, channels="RGB")

        # Optionally, display the input vector used for these images
        st.sidebar.subheader("Custom Noise Vector:")
        st.sidebar.write(list(noise_vector[0].cpu().numpy()))





