import os
import cv2
import requests
import threading

api_key = os.getenv("ROBOFLOW_KEY")
upload_url = "https://detect.roboflow.com/vehicle-detection-iusts/2"

def analyze_image_file(image_data, frame):
    response = requests.post(
        upload_url,
        files={"file": image_data},
        params={"api_key": api_key, "name": "vehicle-detection"},
    )
    detections = response.json()

    for detection in detections.get("predictions", []):
        x, y, w, h = (
            detection["x"] - detection["width"] // 2,
            detection["y"] - detection["height"] // 2,
            detection["width"],
            detection["height"],
        )
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            detection["class"],
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (36, 255, 12),
            2,
        )

def analyze_frame_async(frame):
    _, img_encoded = cv2.imencode('.jpg', frame)
    img_bytes = img_encoded.tobytes()
    analyze_image_file(img_bytes, frame)

video_url = "sample_video.mp4"

cap = cv2.VideoCapture(video_url)

if not cap.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

frame_skip = 1
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    frame_resized = cv2.resize(frame, (640, 360))

    if frame_count % frame_skip == 0:
        threading.Thread(target=analyze_frame_async, args=(frame_resized,)).start()

    cv2.imshow('Video', frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
