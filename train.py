import streamlit as st

def app():
    st.title("Hyper Parameter")

    train_model = st.sidebar.selectbox("Select Model", ["GAN", "DCGAN"])
    if train_model == "GAN":
        st.write("Welcome GAN")
    elif train_model == "DCGAN":
        train_option = st.sidebar.selectbox("Select Option", ["Model", "Train Model", "Remodel"])
        if train_option == "Model":
            st.write("Welcome DCGAN MODEL")
        elif train_option == "Remodel":
            st.write("Welcome DCGAN REMODEL")
        elif train_option == "Train Model":
            st.write("Welcome TRAIN DCGAN MODEL")
 
if __name__ == "__main__":
    app()
