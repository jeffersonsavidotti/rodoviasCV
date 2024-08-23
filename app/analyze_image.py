import requests
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np

# Configurações do recurso de previsão
url = "https://rodoviascv-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/3ebe3ea3-c5e9-4c04-8c2f-7a4cadb4eea7/detect/iterations/Iteration3/image"
prediction_key = "a3de1de0129c45fcb155531060227b37"
headers = {
    "Prediction-Key": prediction_key,
    "Content-Type": "application/octet-stream"
}

# Carrega a imagem de teste
img_file = 'Images/Test/test (1).jpg'

# Envia a solicitação
with open(img_file, 'rb') as image_data:
    response = requests.post(url, headers=headers, data=image_data)

# Verifica a resposta
if response.status_code == 200:
    results = response.json()
else:
    print("Erro:", response.status_code, response.text)
    exit()

# Cria uma imagem a partir da resposta
img = Image.open(img_file)
img_width, img_height = img.size

# Cria uma figura para exibir os resultados
draw = ImageDraw.Draw(img)

# Configura a largura da linha e as cores dos objetos
lineWidth = int(img_width / 100)
object_colors = {
    "carro": "yellow",
    "caminhao": "cyan",
    "onibus": "green",
    "moto": "blue",
    "van": "orange"
}

# Exibe os resultados
for prediction in results.get("predictions", []):
    if prediction["probability"] * 100 > 50:  # Apenas exibe se a probabilidade for maior que 50%
        tag_name = prediction["tagName"]
        color = object_colors.get(tag_name, 'white')  # Usa 'white' como cor padrão
        left = prediction["boundingBox"]["left"] * img_width
        top = prediction["boundingBox"]["top"] * img_height
        width = prediction["boundingBox"]["width"] * img_width
        height = prediction["boundingBox"]["height"] * img_height
        right = left + width
        bottom = top + height
        
        # Cria um retângulo
        draw.rectangle([left, top, right, bottom], outline=color, width=lineWidth)
        
        # Adiciona anotações
        draw.text((left, top), f"{tag_name}: {prediction['probability'] * 100:.2f}%", fill='white')

# Salva a imagem com as anotações em um arquivo
output_file = 'Images/Train/test (1)_annotated.jpg'
img.save(output_file)
print(f"Imagem anotada salva como {output_file}")

