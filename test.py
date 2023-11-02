import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import base64
import plotly.express as px
dim_z = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ngpu = 1

df = px.data.iris()

@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


img = get_img_as_base64("./Image/background.jpg")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
    background-image: url("https://imageio.forbes.com/specials-images/imageserve/64aa09f4d3dcc2e90091cf1f/Abstract-dots-and-waves-on-black-background-showing-how-AI-will-change-work-/960x0.jpg?format=jpg&width=1440");
    background-size: 100%;
    background-position: top left;
    background-repeat: no-repeat;
    background-attachment: local;
}}

[data-testid="stSidebar"] > div:first-child {{
    background-image: url("data:image/png;base64,{img}");
    background-position: center; 
    background-repeat: no-repeat;
    background-attachment: fixed;
}}

[data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
    right: 2rem;
}}
.container {{
    padding: 2rem;
    text-align: center;
.title {{
    font-size: 40px;
    color: black;
    margin-bottom: 2rem;
}}
.h1 {{
    color: black;
}}
.subtitle {{
    font-size: 30px;
    color: black;
    margin-bottom: 2rem;
    text-align: left;
}}
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

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

model_G = Generator(ngpu=1)
model_G.to(device)
# Define paths to pre-trained models
path_train = "./Model/celeba_gen_train.pth"
path_retrain = "./Model/generator_retrained.pth"
path_model_train = "./Model/celeba_gen_1.pth"
from test import Generator

# Load pre-trained models and set to evaluation mode
def load_and_evaluate_model(path):
    model_instance = Generator(ngpu=1)
    model_instance.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model_instance = model_instance.eval()
    return model_instance

model_train = load_and_evaluate_model(path_train)
model_retrain = load_and_evaluate_model(path_retrain)
model_model_train = load_and_evaluate_model(path_model_train)
def app():
    # Create a Streamlit app
    st.header("DCGAN Image Generator")
    # model_options = ["-- Choose processing model --","Train", "RETRAIN", "ModelTrain"]
    # selected_option = st.sidebar.selectbox("Select a model", model_options)
    # Sidebar to select the model
    selected_model = None
    train_model = st.sidebar.selectbox("Select Model", ["GAN", "DCGAN"])
    if train_model == "GAN":
        st.write("Welcome GAN")
    elif train_model == "DCGAN":
        train_option = st.sidebar.selectbox("Select Option", ["Model", "Train Model", "Remodel"])
        if train_option == "Model":
            selected_model = model_model_train
        elif train_option == "Remodel":
            selected_model = model_retrain
        elif train_option == "Train Model":
            selected_model = model_train
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
