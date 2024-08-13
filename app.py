import streamlit as st
import cv2
import numpy as np
import easyocr
import torch
from matplotlib import pyplot as plt
from PIL import Image
from torchvision.transforms import ToTensor

# Import the Generator class from model.py
from model import Generator

# Load the EasyOCR reader
reader = easyocr.Reader(['en'])

def extract_text(image):
    results = reader.readtext(image)
    bounding_boxes = []
    texts = []
    for (bbox, text, prob) in results:
        bounding_boxes.append(bbox)
        texts.append(text)
    return bounding_boxes, texts

def create_mask(image, bounding_boxes):
    mask = np.zeros(image.shape[:2], dtype="uint8")
    for bbox in bounding_boxes:
        pts = np.array([bbox[0], bbox[1], bbox[2], bbox[3]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)
    return mask

def inpaint_image(image, mask, model):
    image_tensor = ToTensor()(image).unsqueeze(0)
    mask_tensor = ToTensor()(mask).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor, mask_tensor)
    
    inpainted_image = output.squeeze(0).permute(1, 2, 0).numpy()
    inpainted_image = (inpainted_image * 255).astype(np.uint8)
    
    return inpainted_image

def remove_text(image, bounding_boxes):
    no_text_image = image.copy()
    for bbox in bounding_boxes:
        pts = np.array([bbox[0], bbox[1], bbox[2], bbox[3]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(no_text_image, [pts], (255, 255, 255))
    return no_text_image

def process_image(image, model):
    # Convert PIL image to OpenCV format
    image = np.array(image.convert("RGB"))
    
    # Extract text and bounding boxes
    bounding_boxes, texts = extract_text(image)
    
    # Create mask for the text regions
    mask = create_mask(image, bounding_boxes)
    
    # Inpaint the image
    inpainted_image = inpaint_image(image, mask, model)
    
    # Create an image with text regions removed (filled with white)
    no_text_image = remove_text(image, bounding_boxes)
    
    # Draw bounding boxes on the original image
    for bbox in bounding_boxes:
        cv2.polylines(image, [np.array(bbox, np.int32)], True, (0, 255, 0), 2)

    return image, mask, no_text_image, inpainted_image, texts

# Load the DeepFillv2 model
model = Generator()
model.eval()

# Streamlit UI
st.title("Text Removal from Images")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Process the image
    original_image, mask, no_text_image, inpainted_image, extracted_texts = process_image(image, model)
    
    # Display results in a single row
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(original_image, caption="Original Image with Text Bounding Boxes", use_column_width=True)
    with col2:
        st.image(mask, caption="Text Mask", use_column_width=True, channels="GRAY")
    with col3:
        st.image(no_text_image, caption="Image with Text Removed (White Fill)", use_column_width=True)

    
    # Display the extracted text
    st.write("### Extracted Texts:")
    for i, text in enumerate(extracted_texts):
        st.write(f"Text {i+1}: {text}")
