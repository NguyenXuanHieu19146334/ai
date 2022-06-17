import time
import numpy as np
import streamlit as st
import tensorflow as tf
from streamlit_option_menu import option_menu
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input

model = tf.keras.models.load_model("hieu.h5")

st.header('Nhận diện tỉ lệ giống nhau của diễn viên')
st.subheader('Nguyễn Xuân Hiếu 19146334')
### load file
uploaded_file = st.file_uploader("Choose an image file", type=["jpg","jpeg","png"])
    
map_dict = {0: 'bill_gates',
                1: 'elon_musk',
                2:'jeff_bezos',
                3: 'mark_zuckerberg',
                4:'steve_jobs'}
    
if uploaded_file is not None:
    # Convert the file
    img = image.load_img(uploaded_file,target_size=(70,70))
    st.image(uploaded_file, channels="RGB")
    img = img_to_array(img)
    img = img.reshape(1,70,70,3)
    img = img.astype('float32')
    img = img/255
      
    Genrate_pred = st.button("Generate Prediction") 
    
    if Genrate_pred:
         prediction = model.predict(img).argmax()
         st.title("Predicted Label for the image is {}".format(map_dict [prediction]))
         y_pred = model.predict(img)             
         a = y_pred.max()
         a = a*100
         st.write("**Accuracy:** ",a,"%")
        
