import streamlit as st
import torch
import torch.nn as nn
from keras.models import load_model
from io import StringIO

def display_hyperparameters(model,model_name):
    st.header(f"Parameter {model_name}:")
    for name, param in model.named_parameters():
        st.write(f"Parameter name: {name}, Size: {param.size()}")
path_train = "./Model/celeba_gen_train.pth"
path_retrain = "./Model/generator_retrained.pth"
path_model_train = "./Model/celeba_gen_1.pth"

path_train_Dis = "./Model/celeba_ds_train.pth"
path_reatrain_Dis = "./Model/discriminator_trained.pth"
path_model_train_Dis = "./Model/celeba_ds_1.pth"

from test import Generator
from test import Discriminator
# Load pre-trained models and set to evaluation mode
def load_and_evaluate_model(path):
    model_instance = Generator(ngpu=1)
    model_instance.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model_instance = model_instance.eval()
    return model_instance

def load_and_evaluate_model_Dis(path):
    model_instance = Discriminator(ngpu=1)
    model_instance.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model_instance = model_instance.eval()
    return model_instance
# Load the saved GAN model
loaded_gan = load_model('./Model/gan_model_hyper.h5')

buffer = StringIO()
loaded_gan.summary(print_fn=lambda x: buffer.write(x + '\n'))
model_summary = buffer.getvalue()
# Print optimizer configuration
optimizer_config = loaded_gan.optimizer.get_config()
model_train = load_and_evaluate_model(path_train)
model_retrain = load_and_evaluate_model(path_retrain)
model_model_train = load_and_evaluate_model(path_model_train)

model_train_Dis = load_and_evaluate_model_Dis(path_train_Dis)
model_retrain_Dis = load_and_evaluate_model_Dis(path_reatrain_Dis)
model_model_train_Dis = load_and_evaluate_model_Dis(path_model_train_Dis)

def app():
    st.title("Hyper Parameter")

    train_model = st.sidebar.selectbox("Select Model", ["GAN", "DCGAN"])
    if train_model == "GAN":
        st.write("Kích thước (chiều rộng và chiều cao) của ảnh đầu vào: 64")
        st.write("Kích thước batch (số lượng mẫu dữ liệu):  16")
        st.write("Số lượng luồng làm việc (workers) được sử dụng để nạp dữ liệu trong quá trình đào tạo:    1")
        st.write("Số chiều của vector nhiễu (latent vector)sử dụng làm đầu vào cho mô hình Generator: 32")
        st.write("Tốc độ học: 0.0001")
        st.write("Giới hạn giá trị của các gradient: clipvalue=1.0")
        st.header("Loaded GAN Model Summary")
        st.text(model_summary)
        st.header("Optimizer Configuration:")
        st.write(optimizer_config)
    elif train_model == "DCGAN":
        train_option = st.sidebar.selectbox("Select Option", ["Model", "Train Model", "Remodel"])
        if train_option == "Model":
            st.write("Kích thước (chiều rộng và chiều cao) của ảnh đầu vào: 64")
            st.write("Kích thước batch (số lượng mẫu dữ liệu):  128")
            st.write("Số lượng luồng làm việc (workers) được sử dụng để nạp dữ liệu trong quá trình đào tạo:    1")
            st.write("Số chiều của vector nhiễu (latent vector)sử dụng làm đầu vào cho mô hình Generator: 100")
            st.write("Số vòng lặp dùng để training(Num_Epoch): 30")
            st.write("Tốc độ học: 0.0002")
            st.write("Tham số beta của thuật toán tối ưu hóa Adam: (0.5, 0.999)")
            display_hyperparameters(model_train, "Generator")
            display_hyperparameters(model_train_Dis, "Discriminator")
        elif train_option == "Remodel":
            st.write("Kích thước (chiều rộng và chiều cao) của ảnh đầu vào: 64")
            st.write("Kích thước batch (số lượng mẫu dữ liệu):128")
            st.write("Số lượng luồng làm việc (workers) được sử dụng để nạp dữ liệu trong quá trình đào tạo:    1")
            st.write("Số chiều của vector nhiễu (latent vector)sử dụng làm đầu vào cho mô hình Generator: 100")
            st.write("Số vòng lặp dùng để training(Num_Epoch): 30")
            st.write("Tốc độ học: 0.0002")
            st.write("Tham số beta của thuật toán tối ưu hóa Adam: (0.5, 0.999)")
            display_hyperparameters(model_retrain, "Generator")
            display_hyperparameters(model_retrain_Dis, "Discriminator")
        elif train_option == "Train Model":
            st.write("Kích thước (chiều rộng và chiều cao) của ảnh đầu vào: 64")
            st.write("Kích thước batch (số lượng mẫu dữ liệu):  128")
            st.write("Số lượng luồng làm việc (workers) được sử dụng để nạp dữ liệu trong quá trình đào tạo:    1")
            st.write("Số chiều của vector nhiễu (latent vector)sử dụng làm đầu vào cho mô hình Generator: 100")
            st.write("Số vòng lặp dùng để training(Num_Epoch): 50")
            st.write("Tốc độ học: 0.0005")
            st.write("Tham số beta của thuật toán tối ưu hóa Adam: (0.5, 0.999)")
            display_hyperparameters(model_model_train, "Generator")
            display_hyperparameters(model_model_train_Dis, "Discriminator")
 
if __name__ == "__main__":
    app()
