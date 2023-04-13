import streamlit as st
import cv2
from PIL import Image
import numpy as np
import os
import sys
from glob import glob
import requests
from io import BytesIO

from const import CLASSES, COLORS
from settings import DEFAULT_CONFIDENCE_THRESHOLD, DEMO_IMAGE, MODEL, PROTOTXT


@st.cache_data()
def process_image(image):
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5
    )
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    net.setInput(blob)
    detections = net.forward()
    return detections


@st.cache_data()
def annotate_image(
    image, detections, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD
):
    # loop over the detections
    (h, w) = image.shape[:2]
    labels = []
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_threshold:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # display the prediction
            label = f"{CLASSES[idx]}: {round(confidence * 100, 2)}%"
            labels.append(label)
            cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(
                image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2
            )
    return image, labels
# =======
#   App
# =======
st.title("Object detection with MobileNet SSD")
sys.path.insert(0, ".")
st.write(*CLASSES)

# provide options to either select an image form the gallery, upload one, or fetch from URL
gallery_tab, upload_tab, url_tab = st.tabs(["Gallery", "Upload", "Image URL"])

with gallery_tab:
    gallery_files = glob(os.path.join(".", "images", "*"))
    gallery_dict = {image_path.split("/")[-1].split(".")[-2]: image_path
        for image_path in gallery_files}
    st.write(gallery_dict.keys())
    options = list(gallery_dict.keys())
    file_name = st.selectbox("Select Art", 
                        options=options, index=options.index("demo"))
    
    file = '.' + gallery_dict[file_name]
    if st.session_state.get("file_uploader") is not None:
        st.warning("To use the Gallery, remove the uploaded image first.")
    if st.session_state.get("image_url") not in ["", None]:
        st.warning("To use the Gallery, remove the image URL first.")
    st.write(file)
    image = Image.open(file)

with upload_tab:
    img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))
    
    #else:
    #    demo_image = DEMO_IMAGE
    #    image = np.array(Image.open(demo_image))

with url_tab:
    url_text = st.empty()
    url_reset = st.button("Clear URL", key="url_reset")
    if url_reset and "image_url" in st.session_state:
        st.session_state["image_url"] = ""
        st.write(st.session_state["image_url"])

    url = url_text.text_input("Image URL", key="image_url")
    
    if url!="":
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content))
        except:
            st.error("The URL does not seem to be valid.")

confidence_threshold = st.slider(
    "Confidence threshold", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05
)
detections = process_image(image)
image, labels = annotate_image(image, detections, confidence_threshold)

st.image(
    image, caption=f"Processed image", use_column_width=True,
)

st.write(labels)
