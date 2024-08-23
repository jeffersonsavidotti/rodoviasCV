from dotenv import load_dotenv
import requests
import cv2
from PIL import Image
import numpy as np
import io
import os
import streamlit as st
from tempfile import NamedTemporaryFile
import time

load_dotenv()

# Carrega a chave de predição do arquivo .env
prediction_key = os.getenv("PREDICTION_KEY")

# Modelos treinados
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

headers_image = {
    "Prediction-Key": prediction_key,
    "Content-Type": "application/octet-stream"
}

headers_url = {
    "Prediction-Key": prediction_key,
    "Content-Type": "application/json"
}

# Função para detectar objetos em uma imagem ou URL
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

# Função para formatar a descrição dos objetos detectados
def format_predictions(predictions, threshold):
    description = ""
    for prediction in predictions:
        probability = prediction['probability']
        if probability >= threshold:
            description += f"{prediction['tagName']}: {probability * 100:.2f}%\n"
    return description.strip()

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

    # Adiciona a descrição dos objetos detectados
    st.write("Descrição das Detecções:")
    st.text(format_predictions(results.get("predictions", []), threshold))

    # Limpa arquivos temporários
    os.remove(temp_input_path)
    os.remove(temp_output_path)

# Função para processar e exibir imagens com Streamlit
def process_and_display_image(uploaded_file=None, image_url=None, model="Iteration3", threshold=0.5):
    if uploaded_file:
        image_data = uploaded_file.read()
        is_url = False
    elif image_url:
        response = requests.get(image_url)
        image_data = response.content
        is_url = True
    else:
        st.error("Nenhum arquivo ou URL fornecido.")
        return
    
    # Detecta objetos na imagem
    results = detect_objects(image_data, model, is_url, image_url)
    if results:
        img = Image.open(io.BytesIO(image_data))
        frame = np.array(img)

        # Desenha as anotações na imagem
        frame_with_annotations = draw_predictions_on_frame(frame, results.get("predictions", []), threshold)

        # Exibe a imagem original e a anotada
        st.image(img, caption="Imagem Original", use_column_width=True)
        st.image(frame_with_annotations, caption="Imagem com Anotações", use_column_width=True)
        
        # Adiciona a descrição dos objetos detectados
        st.write("Descrição das Detecções:")
        st.text(format_predictions(results.get("predictions", []), threshold))

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
            st.success(f"Modelo {new_model_name} adicionado com sucesso!")
            selected_model = new_model_name
        else:
            st.error("Todos os campos são obrigatórios para adicionar um novo modelo.")
elif selected_model not in models:
    st.error("Modelo selecionado não encontrado.")
    st.stop()

st.write(f"Modelo selecionado: {selected_model}")

# Escolha entre imagem e vídeo
option = st.selectbox("Escolha entre carregar uma imagem ou vídeo", ["Imagem", "Vídeo"])

# Parâmetros de detecção
threshold = st.slider("Defina o limiar de confiança", 0.0, 1.0, 0.5)

if option == "Imagem":
    uploaded_image = st.file_uploader("Carregar uma imagem", type=["jpg", "jpeg", "png"])
    image_url = st.text_input("ou insira a URL da imagem")
    if st.button("Processar Imagem"):
        process_and_display_image(uploaded_file=uploaded_image, image_url=image_url, model=selected_model, threshold=threshold)
elif option == "Vídeo":
    uploaded_video = st.file_uploader("Carregar um vídeo", type=["mp4", "avi", "mov"])
    video_url = st.text_input("ou insira a URL do vídeo")
    num_frames = st.slider("Número de frames a processar", 1, 500, 100)
    if st.button("Processar Vídeo"):
        process_and_display_video(uploaded_file=uploaded_video, video_url=video_url, model=selected_model, num_frames=num_frames, threshold=threshold)
