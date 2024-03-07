import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import plotly.express as px
import time
from keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import functools
import torch.nn as nn
import torch.nn.functional as F

dim_z = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ngpu = 1

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)
class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)
    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out
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
path_model_train = "./Model/celeba_gen_1.pth"
def load_and_evaluate_model(path):
    model_instance = Generator(ngpu=1)
    model_instance.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model_instance = model_instance.eval()
    return model_instance
model_model_train = load_and_evaluate_model(path_model_train)
import streamlit as st

def app():
    st.header("DCGAN Image Generator")
    selected_model = None
    selected_model = model_model_train
    if selected_model is not None:
        st.sidebar.header("Customize Noise Vector")
        if "noise_vector" not in st.session_state:
            st.session_state.noise_vector = torch.randn(1, dim_z, 1, 1, device=device)
        noise_vector_widget = st.sidebar.slider("Adjust Noise Vector", -3.0, 3.0, 0.0)
        st.session_state.noise_vector += noise_vector_widget
        with torch.no_grad():
            generated_images = selected_model(st.session_state.noise_vector)
        generated_images = (generated_images + 1) / 2
        model_path = './Model/RRDB_ESRGAN_x4.pth'  # Thay thế bằng đường dẫn tới mô hình
        model = RRDBNet(3, 3, 64, 23, gc=32).to(device)
        model.load_state_dict(torch.load(model_path), strict=True)
        model.eval()
        upscale = st.sidebar.checkbox('Upscale image', value= False)
        if upscale:
            with torch.no_grad():
                output = model(generated_images).data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = output.transpose(1, 2, 0)
            output = (output * 255.0).round()
            output = output.astype(np.uint8)
        else:
            output = generated_images.squeeze().permute(1, 2, 0).cpu().numpy()
        st.image(output, caption='Generated Image')
        st.sidebar.subheader("Custom Noise Vector:")
        st.sidebar.write(list(st.session_state.noise_vector[0].cpu().numpy()))


