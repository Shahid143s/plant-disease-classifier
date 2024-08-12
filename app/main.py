import streamlit as st
import os
import tensorflow as tf

import json
from PIL import Image
import pandas as pd
import numpy as np





st.markdown("""
    <style>
        .subheader-box {
            background-color: #F5F5F5;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
            margin-bottom: 2rem;
            width: 80%;
            text-align: center;
        }
        .subheader-text {
            font-size: 1.2em;
            color: #31333F;
        }
    </style>
    """, unsafe_allow_html=True)

# Load the pre-trained model and class names
working_dir = os.path.dirname(os.path.abspath(__file__))
#Dropbox model URL
model_url = "https://www.dropbox.com/scl/fi/o028ea99oh7zgkmm33hs5/plant_disease_prediction_model.h5?rlkey=r2gf56j37gf7eo0ywagodg7i2&st=uxkau5kn&dl=1"

# Download the model file
model_path = os.path.join(working_dir, "plant_disease_prediction_model.h5")
response = requests.get(model_url)
with open(model_path, "wb") as f:
    f.write(response.content)

model = tf.keras.models.load_model(model_path)
class_indices = json.load(open(f"{working_dir}/modified_class_indices.json"))

# Function to load and preprocess the image
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

# Function to predict the class of an image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name



# Set theme and style
st.markdown("""
    <style>
        .main {
            text-align: center;
        }
        .css-18e3th9 {
            background-color: #F5F5F5;
        }
        .css-1v0mbdj {
            background-color: #DCBA00;
            color: #31333F;
            font-size: 1.2em;
        }
        .css-1v0mbdj:hover {
            background-color: #B59B00;
        }
        .css-1nqz20f {
            margin: 1.5rem 0;
        }
        .css-1j3w5l8 {
            color: #FF4742;
            font-size: 1.5em;
            text-align: center;
        }
        .stButton {
            display: block;
            margin-left: auto;
            margin-right: auto;
        }
    </style>
    """, unsafe_allow_html=True)

# Centered layout
st.markdown('<div class="main">', unsafe_allow_html=True)

st.title("Plant Disease Classification")
st.markdown('<div class="title">Upload an image of a <b>supported</b> plant leaf to classify its disease.<br><hr></div>', unsafe_allow_html=True)
st.warning("Supported formats .jpg, .jpeg and .png")

uploaded_image = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="visible")

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    resized_img = image.resize((300, 300))  # Adjust size for better visibility

    # Create a single column layout for centering the image
    col1, col2, col3 = st.columns([2, 4, 1])  # Create columns with proportions

    with col2:  # Center column
        st.image(resized_img, caption='Uploaded Image', use_column_width=False, width=300)
    
    if st.button('Find Disease'):
        # Preprocess the uploaded image and predict the class
        prediction = predict_image_class(model, uploaded_image, class_indices)
        import time
        col1, col2, col3 = st.columns([2, 4, 1])  # Adjust column widths as needed

        with col2:  # Center column
            with st.spinner('Finding Disease...'):
                time.sleep(2)  # Simulate a task
        st.success(f"### {prediction}")
st.markdown('</div>', unsafe_allow_html=True)





st.sidebar.title("Available Plants:")

# Plant data with improved descriptions for diseases
plants_data = {
    "Select Plant": [
        ""
    ],

    "Apple": [
        "Apple Scab", 
        "Black Rot", 
        "Cedar Apple Rust", 
        "Healthy"
    ],
    "Blueberry": [
        "Healthy"
    ],
    "Cherry": [
        "Powdery Mildew", 
        "Healthy"
    ],
    "Corn": [
        "Cercospora Leaf Spot", 
        "Common Rust", 
        "Northern Leaf Blight", 
        "Healthy"
    ],
    "Grape": [
        "Black Rot", 
        "Esca (Black Measles)", 
        "Leaf Blight (Isariopsis Leaf Spot)", 
        "Healthy"
    ],
    "Orange": [
        "Huanglongbing (Citrus Greening)"
    ],
    "Peach": [
        "Bacterial Spot", 
        "Healthy"
    ],
    "Pepper": [
        "Bacterial Spot", 
        "Healthy"
    ],
    "Potato": [
        "Early Blight", 
        "Late Blight", 
        "Healthy"
    ],
    "Raspberry": [
        "Healthy"
    ],
    "Soyabean": [
        "Healthy"
    ],
    "Squash": [
        "Powdery Mildew"
    ],
    "Strawberry": [
        "Leaf Scorch", 
        "Healthy"
    ],
    "Tomato": [
        "Bacterial Spot", 
        "Early Blight", 
        "Late Blight", 
        "Leaf Mold", 
        "Septoria Leaf Spot", 
        "Spider Mites", 
        "Target Spot", 
        "Yellow Leaf Curl Virus", 
        "Mosaic Virus", 
        "Healthy"
    ]
}

# Sidebar elements
st.sidebar.info("Available diseases for detection")

# Dropdown to select plant type
plant_name = st.sidebar.selectbox("Select a plant type:", list(plants_data.keys()))

# Display the diseases for the selected plant type
if plant_name != "Select Plant":
    st.sidebar.subheader(f"Diseases for {plant_name}:")
    diseases = plants_data[plant_name]
    
    # Convert the diseases list to a DataFrame for table display
    diseases_df = pd.DataFrame(diseases, columns=["Disease Description"])
    
    # Display the diseases in a table format
    st.sidebar.table(diseases_df)
else:
    st.sidebar.write("Please select a plant type to see the available diseases.")
