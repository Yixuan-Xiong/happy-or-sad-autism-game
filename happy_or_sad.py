import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
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
if "mode" not in st.session_state:
    st.session_state["mode"] = "home"

if st.session_state["mode"] == "home":
    st.title("🎭 HAPPY OR SAD")
    st.subheader("Select Mode:")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👤 Single-player"):
            st.session_state["mode"] = "single"
            st.rerun()
    
    with col2:
        if st.button("👥 Two-players"):
            st.session_state["mode"] = "double"
            st.rerun()

# Initialize camera
# Ask Chatgpt to use cv2.CAP_AVFOUNDATION can improve mac's camera performance
camera_capture = cv2.VideoCapture(0,cv2.CAP_AVFOUNDATION) 
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

# Two-player mode, detect only 2 face
if st.session_state["mode"] == "double":
    number_of_faces_to_detect = 2
# Single-player mode, detect only 1 face
else:
    number_of_faces_to_detect = 1

# Countdown + real-time score monitoring
def countdown_with_live_monitoring(video_placeholder, score_placeholder):

    with mp_face_mesh.FaceMesh(
        static_image_mode=False, 
        max_num_faces=number_of_faces_to_detect, 
        refine_landmarks=True
    ) as real_time_face_mesh_model:

        timer_placeholder = st.empty()
        captured_frame_rgb = None 
        
        # Ask Chatgpt about the countdown timer design
        start_time = time.time()
        while time.time() - start_time < 5:
            remaining_time = int(5 - (time.time() - start_time))
            timer_placeholder.subheader(f"⏳ Count down: {remaining_time}")
            capture_success, frame = camera_capture.read()

            if capture_success:
                video_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(video_frame_rgb, caption="Live Camera", use_container_width=True)
            else:
                st.warning("Unable to read from camera")
                break

        timer_placeholder.empty()

        # Freeze frame after countdown
        captured_frame_rgb = video_frame_rgb.copy()

        # Detect face landmarks
        face_result = real_time_face_mesh_model.process(video_frame_rgb)

        # Single-player mode 
        if st.session_state["mode"] == "single":
            landmarks = extract_facial_landmarks(video_frame_rgb, real_time_face_mesh_model)
            if landmarks is not None:
                similarity_score = calculate_cosine_similarity(
                    reference_facial_landmarks.flatten(),
                    landmarks.flatten()
                ) * 100
                score_placeholder.write(f"👤 Match Score: **{similarity_score:.1f}%**")
            else:
                score_placeholder.write("👤 Match Score: **--%**")
            video_placeholder.image(video_frame_rgb, channels="RGB")

        # Two-player mode 
        elif st.session_state["mode"] == "double":
            faces = face_result.multi_face_landmarks

            if faces and len(faces) >= 1:
                landmarks_1 = extract_facial_landmarks(video_frame_rgb, real_time_face_mesh_model)
                if landmarks_1 is not None:
                    similarity_score_1 = calculate_cosine_similarity(
                        reference_facial_landmarks.flatten(),
                        landmarks_1.flatten()
                    ) * 100
                    score_placeholder.write(f"👥 Player Match Score: **{similarity_score_1:.1f}%**")
                else:
                    score_placeholder.write("👥 Player Match Score: **--%**")
                video_placeholder.image(video_frame_rgb, channels="RGB")
            else:
                score_placeholder.write("👥 Waiting for Player...")

        # Freeze final image
        if captured_frame_rgb is not None:
            video_placeholder.image(captured_frame_rgb, 
                                    caption="Camera freeze", 
                                    use_container_width=True
                                    )

# Single player mode
if st.session_state["mode"] == "single":
    st.subheader("Please imitate the target's expression")
    st.image(reference_image_rgb, 
             caption="target expression", 
             use_container_width=True)

    video_placeholder = st.empty()
    score_placeholder = st.empty()
    countdown_with_live_monitoring(video_placeholder, score_placeholder)

# Two-player mode
elif st.session_state["mode"] == "double":
    st.subheader("The two players imitate one by one")
    st.image(reference_image_rgb, 
             caption="target expression", 
             use_container_width=True)

    col1, col2 = st.columns(2)

    # player 1
    col1.subheader("player 1")
    video_placeholder_1 = col1.empty()
    score_placeholder_1 = col1.empty()
    countdown_with_live_monitoring(video_placeholder_1, score_placeholder_1)

    # player 2
    col2.subheader("player 2")
    video_placeholder_2 = col2.empty()
    score_placeholder_2 = col2.empty()
    countdown_with_live_monitoring(video_placeholder_2, score_placeholder_2)

camera_capture.release()
cv2.destroyAllWindows()

