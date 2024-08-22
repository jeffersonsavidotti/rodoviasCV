import cv2
import requests
import numpy as np
import threading
import os

# Dados da API Custom Vision
prediction_url = "https://rodoviascv-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/3ebe3ea3-c5e9-4c04-8c2f-7a4cadb4eea7/detect/iterations/modelo1/image"
headers = {
    "Prediction-Key": "a3de1de0129c45fcb155531060227b37",
    "Content-Type": "application/octet-stream"
}

# Diretórios para salvar os frames
frames_dir = "frames"
if not os.path.exists(frames_dir):
    os.makedirs(frames_dir)

# Função para enviar um arquivo de imagem para análise
def analyze_image_file(image_data):
    response = requests.post(prediction_url, headers=headers, data=image_data)
    return response.json()

# Função para processamento assíncrono
def analyze_frame_async(frame, frame_id):
    _, img_encoded = cv2.imencode('.jpg', frame)
    img_bytes = img_encoded.tobytes()
    prediction = analyze_image_file(img_bytes)
    print(f"Frame {frame_id}: {prediction}")

# Função para criar vídeo a partir dos frames salvos
def create_video_from_frames(frame_files, output_video_path, frame_size, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para .mp4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    for frame_file in sorted(frame_files):
        frame = cv2.imread(frame_file)
        out.write(frame)

    out.release()

# URL do vídeo ou caminho para o arquivo de vídeo
video_url = "sample_video.mp4"

# Captura de vídeo com OpenCV
cap = cv2.VideoCapture(video_url)

if not cap.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

frame_skip = 1  # Analisa um frame a cada 5
frame_count = 0
frame_files = []
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Reduz a resolução do frame para acelerar o processamento
    frame_resized = cv2.resize(frame, (640, 360))

    # Salva o frame
    frame_file = os.path.join(frames_dir, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(frame_file, frame_resized)
    frame_files.append(frame_file)

    # Envia o frame para análise a cada N frames
    if frame_count % frame_skip == 0:
        threading.Thread(target=analyze_frame_async, args=(frame_resized, frame_count)).start()

    # Exibe o frame em uma janela
    cv2.imshow('Video', frame_resized)

    # Pressione 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libere recursos e feche janelas
cap.release()
cv2.destroyAllWindows()

# Cria o vídeo a partir dos frames salvos
create_video_from_frames(frame_files, 'output_video.mp4', frame_size)
