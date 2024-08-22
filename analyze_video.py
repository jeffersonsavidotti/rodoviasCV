import cv2
import requests
import numpy as np
import threading

# Dados da API Custom Vision
prediction_url = "https://rodoviascv-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/3ebe3ea3-c5e9-4c04-8c2f-7a4cadb4eea7/detect/iterations/modelo1/image"
headers = {
    "Prediction-Key": "a3de1de0129c45fcb155531060227b37",
    "Content-Type": "application/octet-stream"
}

# Função para enviar um arquivo de imagem para análise
def analyze_image_file(image_data):
    response = requests.post(prediction_url, headers=headers, data=image_data)
    return response.json()

# Função para desenhar as caixas delimitadoras e as tags no frame
def draw_predictions_on_frame(frame, predictions):
    for prediction in predictions:
        # Coordenadas da bounding box
        left = int(prediction['boundingBox']['left'] * frame.shape[1])
        top = int(prediction['boundingBox']['top'] * frame.shape[0])
        width = int(prediction['boundingBox']['width'] * frame.shape[1])
        height = int(prediction['boundingBox']['height'] * frame.shape[0])

        # Desenhar o retângulo
        cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 255, 0), 2)

        # Colocar a tag e a probabilidade no frame
        label = f"{prediction['tagName']} ({prediction['probability']:.2f})"
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

# Função para processamento assíncrono
def analyze_frame_async(frame, frame_resized):
    _, img_encoded = cv2.imencode('.jpg', frame_resized)
    img_bytes = img_encoded.tobytes()
    prediction = analyze_image_file(img_bytes)

    # Desenha as predições no frame original
    frame_with_predictions = draw_predictions_on_frame(frame, prediction['predictions'])
    
    # Exibe o frame com as predições
    cv2.imshow('Video', frame_with_predictions)

# URL do vídeo ou caminho para o arquivo de vídeo
video_url = "sample_video.mp4"

# Captura de vídeo com OpenCV
cap = cv2.VideoCapture(video_url)

if not cap.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

frame_skip = 5  # Analisa um frame a cada 5
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
        threading.Thread(target=analyze_frame_async, args=(frame, frame_resized)).start()

    # Exibe o frame original enquanto o processamento ocorre
    cv2.imshow('Video', frame_resized)

    # Pressione 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libere recursos e feche janelas
cap.release()
cv2.destroyAllWindows()
