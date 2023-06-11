#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from PIL import Image
from operator import index
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import os
import plotly.express as px
from pycaret.regression import setup, compare_models, pull, save_model, load_model
import pandas_profiling
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os 

os.environ['SM_FRAMEWORK'] = 'tf.keras'

from segmentation_models.losses import bce_jaccard_loss,dice_loss
from segmentation_models.metrics import iou_score,f1_score
model=tf.keras.models.load_model('Updated_final_3.h5',custom_objects={'iou_score':iou_score,'f1-score':f1_score})
height=256
width=256

import pandas as pd

# Add custom CSS
# Add sidebar
# Add Font Awesome CSS and JS files
st.markdown(
        """
        <style>
        .stApp {
            background-image: url('https://wallpaperaccess.com/full/2024142.jpg');
            background-repeat: no-repeat;
            background-size: cover;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">', unsafe_allow_html=True)
st.markdown('<style>.markdown-section p { margin-bottom: 2px !important; }</style>', unsafe_allow_html=True)
st.sidebar.title("About")
st.sidebar.write("<h4>What is Virtual Try<span style='color:red;'>ON</span>.ai ??</h4>", unsafe_allow_html=True)
st.sidebar.write("<div style='text-align: justify; width: 100%;'>"
    "<h6>Virtual trial of clothing allows users to digitally try on outfits without physically wearing them. Using augmented reality or computer vision technology, users can see how different clothes fit and look on their virtual avatars or live video feed. This technology provides an interactive and immersive shopping experience, helping users make informed decisions about their purchases and reducing the need for returns. It offers convenience, saves time, and enhances the overall shopping experience by enabling users to visualize and evaluate clothing options accurately.</h6>"
    "</div>", unsafe_allow_html=True)

# Add sidebar content
st.sidebar.subheader("Contact")
st.sidebar.text("Vikas Reddy")
st.sidebar.markdown('<i class="fas fa-envelope"> bijivemulavikas1998@gmail.com</i>', unsafe_allow_html=True)
st.sidebar.text("Charan Reddy")
st.sidebar.markdown('<i class="fas fa-envelope"> charanjeereddy283@gmail.com</i>', unsafe_allow_html=True)
st.sidebar.text("Jayesh Nigam")
st.sidebar.markdown('<i class="fas fa-envelope"> jayeshnigam241@gmail.com</i>', unsafe_allow_html=True)
st.sidebar.text("Shubham Gupta")
# st.sidebar.markdown('<i class="fas fa-envelope"> bijivemulavikas1998@gmail.com</i>', unsafe_allow_html=True)



# Main content
st.markdown("<h1 style='color: black;align:top'>Try<span style='color: red;'>ON</span>.ai</h1>", unsafe_allow_html=True)
st.write("Experience the convenience of virtual clothing try-on and see how different outfits look on you without stepping foot in a dressing room.")
    


def resize_scale(image,height,width):
    return cv2.resize(image,(width,height))/255
def get_class_names(masks,labes):
    lis=np.unique(masks)
    for i in lis:
        print(str(labels.iloc[i].values[0])+' : '+str(i))
def make_mask(prediction):
    pred_mask=pred_mask_ch = tf.math.argmax(prediction, axis=-1)
    return pred_mask
def get_class_mask(mask,m_class,height,width):
    map_img=np.zeros((height,width))
    map_img[mask==m_class]=m_class
    return map_img
def get_original_fit(original,class_img,map_img,height,width):
    img=resize_scale(original,height,width)*255
    map_img=resize_scale(map_img,height,width)
    img[class_img!=0]=0
    res=np.where(img!=0,img,map_img)
    return res

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
        # Read image as PIL object
    pil_image = Image.open(uploaded_file)

        # Convert PIL image to NumPy array
    img = np.array(pil_image)
    frame=resize_scale(img,256,256) # Resize Frame
    #bath_img=np.zeros((32,256,256,3))
    #bath_img[0]=frame
    x=np.zeros((1,256,256,3))
    x[0]=frame
    # Prediction
    mask=model.predict(x,batch_size=1,verbose=0)
    masks=make_mask(mask)

file_upload = st.file_uploader("Choose an texture image", type=["jpg", "jpeg", "png"])

if file_upload is not None:
    # Read image as PIL object
    pil_image = Image.open(file_upload)

    # Convert PIL image to NumPy array
    map_img = np.array(pil_image)
    labels = pd.read_csv(r"C:\Users\VIKAS REDDY\Downloads\data.csv")
    lis = np.unique(masks)
    dc=dict()
    for i in lis:
        dc[i]=labels['label_list'].iloc[i]

    # Create dropdown widget
    selected_lab = st.selectbox('Select Outfit:', list(dc.values()))
    value=[i for i in dc if dc[i]==selected_lab]
    class_img=get_class_mask(masks[0],value[0],height,width)
    res=get_original_fit(frame,class_img,map_img,height,width)
    result=Image.fromarray((res*255).astype(np.uint8))
    st.image(result,caption='Virtual Image')


# with st.sidebar: 
#     st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
#     st.title("Trial.AI")
#     choice = st.radio("Navigation", ["Upload","Profiling","Modelling", "Download"])
#     st.info("This project application helps you build and explore your data.")

# import matplotlib.pyplot as plt

# if choice == "Upload":
#     st.title("User Image")
#     user_file = st.file_uploader("Upload Your Images", type=['png', 'jpg', 'jpeg'], key="user_image")
#     if user_file:
#         img = plt.imread(user_file)

#         # Perform further processing on the resized frame if needed

# def resize_scale(image,height,width):
#     return cv2.resize(image,(width,height))/255
# def get_class_names(masks,labes):
#     lis=np.unique(masks)
#     for i in lis:
#         print(str(labels.iloc[i].values[0])+' : '+str(i))
# def make_mask(prediction):
#     pred_mask=pred_mask_ch = tf.math.argmax(prediction, axis=-1)
#     return pred_mask
# def get_class_mask(mask,m_class,height,width):
#     map_img=np.zeros((height,width))
#     map_img[mask==m_class]=m_class
#     return map_img
# def get_original_fit(original,class_img,map_img_path,height,width):
#     img=resize_scale(original,height,width)*255
#     map_img=plt.imread(map_img_path)
#     map_img=resize_scale(map_img,height,width)
#     img[class_img!=0]=0
#     res=np.where(img!=0,img,map_img)
#     return res
# frame=resize_scale(img,256,256) # Resize Frame
# #bath_img=np.zeros((32,256,256,3))
# #bath_img[0]=frame
# x=np.zeros((1,256,256,3))
# x[0]=frame

# # Prediction
# mask=model.predict(x,batch_size=1,verbose=0)
# masks=make_mask(mask)
# import matplotlib.pyplot as plt
# import numpy as np

# if choice == "Upload":
#     st.title("Texture Image")
#     texture_file = st.file_uploader("Upload Your Images", type=['png', 'jpg', 'jpeg'], key="texture_image")
#     if texture_file:
#         # Replace the "C:\\Users\\VIKAS REDDY\\Downloads\\fabric-texture-5322099_1280.jpg" path with the uploaded texture_file
#         texture_image = plt.imread(texture_file)
        
#         # Replace the "masks[0]" with your actual masks array
#         class_img = get_class_mask(masks[0], 19, height, width)
        
#         # Replace the "frame" with your actual frame
#         result = get_original_fit(frame, class_img, texture_image, height, width)
        
#         st.image(result)

#         # Display unique values in masks[0] as a dropdown sidebar
#         unique_values = np.unique(masks[0])
#         selected_value = st.sidebar.selectbox("Select a value", unique_values)
#         st.write("Selected value:", selected_value)

#         # Plot the result using plt.imshow
#         plt.imshow(result)
#         st.pyplot()  # Display the plot in Streamlit


