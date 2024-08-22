import requests
import cv2
from PIL import Image
import numpy as np
import io
import os
import streamlit as st
from tempfile import NamedTemporaryFile
import time

# Configurações do recurso de previsão
models = {
    "Iteration3": {
        "url_image": "https://rodoviascv-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/3ebe3ea3-c5e9-4c04-8c2f-7a4cadb4eea7/detect/iterations/Iteration3/image",
        "url_url": "https://rodoviascv-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/3ebe3ea3-c5e9-4c04-8c2f-7a4cadb4eea7/detect/iterations/Iteration3/url"
    },
    "modelo1": {
        "url_image": "https://rodoviascv-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/3ebe3ea3-c5e9-4c04-8c2f-7a4cadb4eea7/detect/iterations/modelo1/image",
        "url_url": "https://rodoviascv-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/3ebe3ea3-c5e9-4c04-8c2f-7a4cadb4eea7/detect/iterations/modelo1/url"
    }
}

prediction_key = "a3de1de0129c45fcb155531060227b37"

headers_image = {
    "Prediction-Key": prediction_key,
    "Content-Type": "application/octet-stream"
}

headers_url = {
    "Prediction-Key": prediction_key,
    "Content-Type": "application/json"
}

# Função para detectar objetos em uma imagem
def detect_objects(image_data, model, is_url=False, url=None):
    model_urls = models[model]
    if is_url:
        response = requests.post(model_urls["url_url"], headers=headers_url, json={"Url": url})
    else:
        response = requests.post(model_urls["url_image"], headers=headers_image, data=image_data)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Erro: {response.status_code} - {response.text}")
        return None

# Função para converter cores nomeadas para BGR
def color_to_bgr(color_name):
    color_dict = {
        "yellow": (0, 255, 255),
        "cyan": (255, 255, 0),
        "green": (0, 255, 0),
        "blue": (255, 0, 0),
        "orange": (0, 165, 255),
        "white": (255, 255, 255)
    }
    return color_dict.get(color_name, (255, 255, 255))

# Função para desenhar anotações no frame
def draw_predictions_on_frame(frame, predictions, threshold):
    object_colors = {
        "carro": "yellow",
        "caminhao": "cyan",
        "onibus": "green",
        "moto": "blue",
        "van": "orange"
    }
    
    for prediction in predictions:
        probability = prediction['probability']
        if probability < threshold:
            continue
        
        left = int(prediction['boundingBox']['left'] * frame.shape[1])
        top = int(prediction['boundingBox']['top'] * frame.shape[0])
        width = int(prediction['boundingBox']['width'] * frame.shape[1])
        height = int(prediction['boundingBox']['height'] * frame.shape[0])

        color_name = object_colors.get(prediction['tagName'], 'white')
        color = color_to_bgr(color_name)
        cv2.rectangle(frame, (left, top), (left + width, top + height), color, 2)
        label = f"{prediction['tagName']} ({probability:.2f})"
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame

# Função para processar e exibir vídeos com Streamlit
def process_and_display_video(uploaded_file=None, video_url=None, model="Iteration3", num_frames=100, threshold=0.5):
    if uploaded_file:
        video_data = uploaded_file.read()
        is_url = False
    elif video_url:
        response = requests.get(video_url)
        video_data = response.content
        is_url = True
    else:
        st.error("Nenhum arquivo ou URL fornecido.")
        return

    # Cria um arquivo temporário para o vídeo
    with NamedTemporaryFile(delete=False, suffix='.mp4') as temp_input_file:
        temp_input_file.write(video_data)
        temp_input_path = temp_input_file.name
    
    cap = cv2.VideoCapture(temp_input_path)
    if not cap.isOpened():
        st.error("Erro ao abrir o vídeo.")
        return

    # Prepara o escritor de vídeo para o arquivo de saída
    fourcc = cv2.VideoWriter_fourcc(*'vp80')  # Codec VP8 para WebM
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    with NamedTemporaryFile(delete=False, suffix='.webm') as temp_output_file:
        temp_output_path = temp_output_file.name
        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
        
        # Total de frames para a barra de progresso
        total_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), num_frames)
        progress_bar = st.progress(0)
        start_time = time.time()

        for _ in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Converte o frame para imagem PIL
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            img_data = buffered.getvalue()
            
            # Detecta objetos
            results = detect_objects(img_data, model, is_url, video_url)
            if results:
                # Desenha as anotações no frame
                frame_with_annotations = draw_predictions_on_frame(frame, results.get("predictions", []), threshold)
                out.write(frame_with_annotations)
            
            # Atualiza a barra de progresso
            progress = int((cap.get(cv2.CAP_PROP_POS_FRAMES) / total_frames) * 100)
            progress_bar.progress(progress)

        cap.release()
        out.release()
        
        # Mostra tempo estimado de conclusão
        elapsed_time = time.time() - start_time
        st.write(f"Tempo decorrido: {elapsed_time:.2f} segundos")
        
    # Exibe o vídeo original e anotado lado a lado
    st.video(temp_input_path, format="video/mp4", start_time=0)
    st.video(temp_output_path, format="video/webm", start_time=0)

    # Limpa arquivos temporários
    os.remove(temp_input_path)
    os.remove(temp_output_path)

# Interface do Streamlit
st.title("Analisador de Imagens e Vídeos com Detecção de Objetos")

# Opção para selecionar o modelo
selected_model = st.selectbox("Escolha o modelo", list(models.keys()) + ["Adicionar Novo Modelo"])

# Se a opção de adicionar novo modelo for selecionada
if selected_model == "Adicionar Novo Modelo":
    new_model_name = st.text_input("Nome do Novo Modelo")
    new_model_image_url = st.text_input("URL do Endpoint de Imagem")
    new_model_url_url = st.text_input("URL do Endpoint de URL")
    if st.button("Adicionar Modelo"):
        if new_model_name and new_model_image_url and new_model_url_url:
            models[new_model_name] = {
                "url_image": new_model_image_url,
                "url_url": new_model_url_url
            }
            st.success("Novo modelo adicionado com sucesso!")
        else:
            st.error("Preencha todos os campos para adicionar o modelo.")
else:
    # Opção para selecionar o tipo de mídia
    option = st.selectbox("Selecione o tipo de mídia", ["Vídeo", "Imagem"])
    
    if option == "Vídeo":
        uploaded_file = st.file_uploader("Escolha um arquivo de vídeo", type=["mp4", "avi", "mov"])
        video_url = st.text_input("Ou insira uma URL de vídeo")
        num_frames = st.slider("Escolha a quantidade de frames a serem processados", min_value=1, max_value=500, value=100)
        threshold = st.slider("Escolha o valor do Threshold da precisão", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        if st.button("Analisar Vídeo"):
            process_and_display_video(uploaded_file, video_url, selected_model, num_frames, threshold)
    elif option == "Imagem":
        uploaded_file = st.file_uploader("Escolha um arquivo de imagem", type=["jpg", "jpeg", "png"])
        image_url = st.text_input("Ou insira uma URL de imagem")
        threshold = st.slider("Escolha o valor do Threshold da precisão", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        if st.button("Analisar Imagem"):
            process_and_display_image(uploaded_file, image_url, selected_model, threshold)
