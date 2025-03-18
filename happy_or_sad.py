import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import time
import os
import random

# Target Emoji Pictures folder
IMAGE_FOLDER_PATH = "/Users/xiongyixuan/Desktop/ai code/AI-4-Media-Project-Yixuan-Xiong/target_images"

# Get all images
image_file_list = [
    file_name for file_name in os.listdir(IMAGE_FOLDER_PATH) 
    if file_name.endswith(('.png', '.jpg', '.jpeg'))
]

# Make sure the folder is not empty
if not image_file_list:
    st.error(f"Target image not found in {IMAGE_FOLDER_PATH} directory, please add at least one emoticon image.")
    st.stop()

# Randomly select a target image
REFERENCE_IMAGE_PATH = os.path.join(
    IMAGE_FOLDER_PATH, random.choice(image_file_list)
)
print(f"Target emoji images: {REFERENCE_IMAGE_PATH}")

# Reads the target image and transfer it to RGB.
reference_image = cv2.imread(REFERENCE_IMAGE_PATH)
if reference_image is None:
    st.error(f"Cannot read the target emoticon, please check if the path is correct.: {REFERENCE_IMAGE_PATH}")
    st.stop()

reference_image_rgb = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)

# Streamlit UI design
st.title("🎭 HAPPY OR SAD")
st.write("Please imitate the target expression")

# Show target images
st.subheader("target expression")

st.image(
    image=reference_image_rgb,  
    caption="target expression",  
    use_container_width=True  
)

# Initialize camera
camera_capture = cv2.VideoCapture(0)
if not camera_capture.isOpened():
    st.error("Cannot open the camera, please check the device permissions.")
    st.stop()

# Use the keypoint Detection Model
mp_face_mesh = mp.solutions.face_mesh
face_mesh_model = mp_face_mesh.FaceMesh(
    static_image_mode=True,  
    max_num_faces=1,         
    refine_landmarks=True,  
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

# Using MediaPipe's predefined set of five sensory keypoints
# mouth
MOUTH_INDEX_LIST = []
for connection in mp_face_mesh.FACEMESH_LIPS:
    for index in connection:
        if index not in MOUTH_INDEX_LIST:
            MOUTH_INDEX_LIST.append(index)

# eyes（left and right）
EYES_INDEX_LIST = []
for connection in mp_face_mesh.FACEMESH_LEFT_EYE:
    for index in connection:
        if index not in EYES_INDEX_LIST:
            EYES_INDEX_LIST.append(index)

for connection in mp_face_mesh.FACEMESH_RIGHT_EYE:
    for index in connection:
        if index not in EYES_INDEX_LIST:
            EYES_INDEX_LIST.append(index)

# eyebrows (left and right)
EYEBROWS_INDEX_LIST = []
for connection in mp_face_mesh.FACEMESH_LEFT_EYEBROW:
    for index in connection:
        if index not in EYEBROWS_INDEX_LIST:
            EYEBROWS_INDEX_LIST.append(index)

for connection in mp_face_mesh.FACEMESH_RIGHT_EYEBROW:
    for index in connection:
        if index not in EYEBROWS_INDEX_LIST:
            EYEBROWS_INDEX_LIST.append(index)

# nose
NOSE_INDEX_LIST = []
for connection in mp_face_mesh.FACEMESH_NOSE:
    for index in connection:
        if index not in NOSE_INDEX_LIST:
            NOSE_INDEX_LIST.append(index)

# Feature point weights
WEIGHTS = {
    "mouth": 2.0,
    "eyes": 1.5,
    "eyebrows": 1.8,
    "nose": 1.0
}

# Extract & normalize facial features to ensure that the results are not affected by the shape of the face
def extract_facial_landmarks(image, face_mesh_model):

    # RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    face_detection_results = face_mesh_model.process(image_rgb)
    
    # No face is detected
    if not face_detection_results.multi_face_landmarks:
        return None  

    facial_landmarks = face_detection_results.multi_face_landmarks[0].landmark

    # Combine all required key point indexes (mouth, eyes, eyebrows, nose)
    key_index_list = MOUTH_INDEX_LIST + EYES_INDEX_LIST + EYEBROWS_INDEX_LIST + NOSE_INDEX_LIST

    # Avoid exceeding the index range
    if max(key_index_list) >= len(facial_landmarks):
        return None  
    
    # Extract the (x, y, z) coordinates of all keypoints and convert to a NumPy array
    key_points_array = np.array([
        [facial_landmarks[index].x, 
         facial_landmarks[index].y, 
         facial_landmarks[index].z]  
        for index in key_index_list
    ])

    # Calculate and normalize facial centroids
    # Calculate the minimum coordinates of the key point
    minimum_coordinates = key_points_array.min(axis=0)
    # Calculate the maximum coordinates of the key point
    maximum_coordinates = key_points_array.max(axis=0)
    # Calculate the coordinates of the center point of the face
    center_coordinates = (minimum_coordinates + maximum_coordinates) / 2.0

    # Calculate the left and right width of the face
    face_width = maximum_coordinates[0] - minimum_coordinates[0] 
    # Calculate the height of the face above and below
    face_height = maximum_coordinates[1] - minimum_coordinates[1] 

    # Take the larger value to ensure the stability of the normalization
    face_size = max(face_width, face_height)
    
    if face_size == 0:
        return None

    # Normalized key point coordinates (subtract center point coordinates and divide by face size)
    normalized_key_points = (key_points_array - center_coordinates) / face_size
    
    # Return to normalized keypoints
    return normalized_key_points

# Calculate the key points of the target emoji images
reference_facial_landmarks = extract_facial_landmarks(reference_image_rgb, face_mesh_model)

# If the key point of the target expression cannot be detected, the program is stopped
if reference_facial_landmarks is None:
    st.error("Cannot detect the face in the target expression picture, please change the image.")
    camera_capture.release()
    st.stop()


# Calculating cosine similarity
def calculate_cosine_similarity(vector_a, vector_b):
    # dot product
    dot_product = np.dot(vector_a, vector_b)
    # Calculate the magnitude of vector A
    magnitude_a = np.linalg.norm(vector_a)
    # Calculate the magnitude of vector B
    magnitude_b = np.linalg.norm(vector_b)

    # Avoid divide-by-zero errors
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    
    # cosine similarity
    cosine_similarity = dot_product / (magnitude_a * magnitude_b)

    return cosine_similarity 

# UI Placeholders
real_time_video_placeholder = st.empty()
matching_score_placeholder = st.empty()


# Run Camera Matching
with mp_face_mesh.FaceMesh(
    static_image_mode=False, 
    max_num_faces=1,  
    refine_landmarks=True,  
    min_detection_confidence=0.5,  
    min_tracking_confidence=0.5  
) as real_time_face_mesh_model:

    # camera data reading loop
    while camera_capture.isOpened():
        capture_success, video_frame = camera_capture.read()

        # If the camera data cannot be read, prompt the user and exit the loop
        if not capture_success:
            st.warning("Unable to read data from the camera, check if the camera is working properly.")
            break

        # RGB
        video_frame_rgb = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)

        # Use FaceMesh to process the current frame and extract face keypoints
        current_facial_landmarks = extract_facial_landmarks(video_frame_rgb, real_time_face_mesh_model)

        # If the face keypoints are successfully detected
        if current_facial_landmarks is not None:
            # Calculate the similarity between the current expression and the target expression
            similarity_score = calculate_cosine_similarity(
                # Array of target expressions
                reference_facial_landmarks.flatten(),
                # Array of camera detections
                current_facial_landmarks.flatten()  
            ) * 100  # Converted to percentage format

            # Show the match score
            matching_score_placeholder.write(f"🎯 Match Score: **{similarity_score:.1f}%**")
        else:
            # If no face keypoints are detected
            matching_score_placeholder.write("🎯 Match Score: **--%**")

        # Displaying the camera screen on the Streamlit (live screen)
        real_time_video_placeholder.image(
            # Convert NumPy arrays to image format
            Image.fromarray(video_frame_rgb), 
            caption="Live camera",  
            use_container_width=True  
        )

        # Make the loop pause briefly to reduce CPU usage and make the screen smoother
        time.sleep(0.05)

camera_capture.release()
cv2.destroyAllWindows()