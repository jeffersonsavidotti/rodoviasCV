import cv2
import requests
import threading

# Roboflow API Key e URL
api_key = "339xX6eBrr8nSafWtOy4"
upload_url = "https://detect.roboflow.com/vehicle-detection-iusts/2"

# Função para enviar um frame para a API Roboflow e obter a detecção
def analyze_image_file(image_data, frame):
    response = requests.post(
        upload_url,
        files={"file": image_data},
        params={"api_key": api_key, "name": "vehicle-detection"},
    )
    detections = response.json()

    # Desenha as detecções no frame
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

# Função para processamento assíncrono
def analyze_frame_async(frame):
    _, img_encoded = cv2.imencode('.jpg', frame)
    img_bytes = img_encoded.tobytes()
    analyze_image_file(img_bytes, frame)

# URL do vídeo ou caminho para o arquivo de vídeo
video_url = "sample_video.mp4"

# Captura de vídeo com OpenCV
cap = cv2.VideoCapture(video_url)

if not cap.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

frame_skip = 1  # Analisa um frame a cada 10
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Reduz a resolução do frame para acelerar o processamento
    frame_resized = cv2.resize(frame, (640, 360))

    # Envia o frame para análise a cada N frames
    if frame_count % frame_skip == 0:
        threading.Thread(target=analyze_frame_async, args=(frame_resized,)).start()

    # Exibe o frame em uma janela
    cv2.imshow('Video', frame_resized)

    # Pressione 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libere recursos e feche janelas
cap.release()
cv2.destroyAllWindows()
