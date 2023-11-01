import streamlit as st

def main():
    st.title("Bài toán GAN và DCGAN")

    st.sidebar.header("Trang chính")
    is_training_page = st.sidebar.button("Train")
    is_testing_page = st.sidebar.button("Test")

    if is_training_page:
        train_option = st.radio("Chọn loại mô hình", ["GAN", "DCGAN"])
        if train_option == "GAN":
            st.write("Welcome GAN")
        elif train_option == "DCGAN":
            dcgan_option = st.radio("Chọn tùy chọn DCGAN", ["Model", "Train Model", "Remodel"])
            if dcgan_option == "Model":
                st.write("Welcome DCGAN MODEL")
            elif dcgan_option == "Train Model":
                st.write("Welcome TRAIN DCGAN MODEL")
            elif dcgan_option == "Remodel":
                st.write("Welcome DCGAN REMODEL")

    if is_testing_page:
        st.write("Welcome to the Test page")

if __name__ == "__main__":
    main()
