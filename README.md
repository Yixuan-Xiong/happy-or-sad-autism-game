# HAPPY OR SAD - Emotion Game for Autism Support
HAPPY OR SAD is an interactive facial expression mimicry game built with MediaPipe and Streamlit. Players are challenged to imitate facial expressions shown on the screen, and the system scores them based on facial similarity. It supports both single-player and two-player modes and is designed to be both fun and educational — especially helpful for children or individuals with autism to practice emotion recognition.  

## Project Overview

This game uses the **MediaPipe FaceMesh** model to detect 478 facial landmarks and compare the player's expression to a target image using cosine similarity. The landmarks are normalised for fair scoring regardless of face shape or size. Streamlit is used to create a smooth, interactive interface.  

## 📁 Project Structure

```text
happy_or_sad/
├── happy_or_sad.py           # Main Streamlit app entry point
├── target_image/             # Folder containing expression reference images
├── requirements.txt          # Python dependencies (streamlit, mediapipe, opencv-python, etc.)
└── README.md                 # Project documentation
```

## project display
https://youtu.be/iXdfAWfinkU


# Setup instructions:

Instructions for setting up the conda environment, any files that need downloading, and the specific technical instructions for how to run your code project go here:

```
streamlit run "happy_or_sad.py"
```


IMAGE_FOLDER_PATH = "/Users/xiongyixuan/Desktop/ai code/AI-4-Media-Project-Yixuan-Xiong/target_image" （If can't run， copy the target_image path）

This is the model: mp_face_mesh = mp.solutions.face_mesh. It already in the code, don't need download again!


