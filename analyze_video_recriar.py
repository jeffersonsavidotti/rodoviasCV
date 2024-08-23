from dotenv import load_dotenv
import cv2
import requests
import numpy as np
import threading
import os

load_dotenv()

prediction_key = os.getenv("PREDICTION_KEY")

# Dados da API Custom Vision
prediction_url = "https://rodoviascv-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/3ebe3ea3-c5e9-4c04-8c2f-7a4cadb4eea7/detect/iterations/modelo1/image"
headers = {
    "Prediction-Key": prediction_key,
    "Content-Type": "application/octet-stream"
}

frames_dir = "frames"
if not os.path.exists(frames_dir):
    os.makedirs(frames_dir)

def analyze_image_file(image_data):
    response = requests.post(prediction_url, headers=headers, data=image_data)
    return response.json()

def analyze_frame_async(frame, frame_id):
    _, img_encoded = cv2.imencode('.jpg', frame)
    img_bytes = img_encoded.tobytes()
    prediction = analyze_image_file(img_bytes)
    print(f"Frame {frame_id}: {prediction}")

def create_video_from_frames(frame_files, output_video_path, frame_size, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    for frame_file in sorted(frame_files):
        frame = cv2.imread(frame_file)
        out.write(frame)

    out.release()

video_url = "sample_video.mp4"

cap = cv2.VideoCapture(video_url)

if not cap.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

frame_skip = 1 
frame_count = 0
frame_files = []
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    frame_resized = cv2.resize(frame, (640, 360))

    frame_file = os.path.join(frames_dir, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(frame_file, frame_resized)
    frame_files.append(frame_file)

    if frame_count % frame_skip == 0:
        threading.Thread(target=analyze_frame_async, args=(frame_resized, frame_count)).start()

    cv2.imshow('Video', frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

create_video_from_frames(frame_files, 'video_com_anotacoes.mp4', frame_size)
