import os
from dotenv import load_dotenv
import cv2
import requests
import numpy as np
import threading

load_dotenv()

prediction_key = os.getenv("PREDICTION_KEY")

# Dados da API Custom Vision
prediction_url = "https://rodoviascv-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/3ebe3ea3-c5e9-4c04-8c2f-7a4cadb4eea7/detect/iterations/modelo1/image"
headers = {
    "Prediction-Key": prediction_key,
    "Content-Type": "application/octet-stream"
}

def analyze_image_file(image_data):
    response = requests.post(prediction_url, headers=headers, data=image_data)
    return response.json()

def draw_predictions_on_frame(frame, predictions):
    for prediction in predictions:
        left = int(prediction['boundingBox']['left'] * frame.shape[1])
        top = int(prediction['boundingBox']['top'] * frame.shape[0])
        width = int(prediction['boundingBox']['width'] * frame.shape[1])
        height = int(prediction['boundingBox']['height'] * frame.shape[0])

        cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 255, 0), 2)

        label = f"{prediction['tagName']} ({prediction['probability']:.2f})"
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

def analyze_frame_async(frame, frame_resized):
    _, img_encoded = cv2.imencode('.jpg', frame_resized)
    img_bytes = img_encoded.tobytes()
    prediction = analyze_image_file(img_bytes)

    frame_with_predictions = draw_predictions_on_frame(frame, prediction['predictions'])
    
    cv2.imshow('Video', frame_with_predictions)

video_url = "sample_video.mp4"

cap = cv2.VideoCapture(video_url)

if not cap.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

frame_skip = 5 
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    frame_resized = cv2.resize(frame, (640, 360))

    if frame_count % frame_skip == 0:
        threading.Thread(target=analyze_frame_async, args=(frame, frame_resized)).start()

    cv2.imshow('Video', frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
