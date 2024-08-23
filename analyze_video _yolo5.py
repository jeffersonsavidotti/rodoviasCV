import torch
import cv2
import argparse

#modelo YOLOv5 pré-treinado
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


vehicle_classes = [2, 3, 5, 7]  # Carro, Moto, Ônibus, Caminhão

def detect_vehicles_in_image(image_path):

    img = cv2.imread(image_path)
    
    results = model(img)
    
    vehicles = results.xyxy[0].numpy()
    for *box, conf, cls in vehicles:
        if int(cls) in vehicle_classes:
            x1, y1, x2, y2 = map(int, box)
            label = f'{model.names[int(cls)]} {conf:.2f}'
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    cv2.imshow('Vehicle Detection - Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_vehicles_in_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)
        
        vehicles = results.xyxy[0].numpy()
        for *box, conf, cls in vehicles:
            if int(cls) in vehicle_classes:
                x1, y1, x2, y2 = map(int, box)
                label = f'{model.names[int(cls)]} {conf:.2f}'
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        cv2.imshow('Vehicle Detection - Video', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="YOLOv5 Vehicle Detection Script")
    parser.add_argument('--image', type=str, help="Caminho para a imagem")
    parser.add_argument('--video', type=str, help="Caminho para o vídeo")

    args = parser.parse_args()

    if args.image:
        detect_vehicles_in_image(args.image)
    elif args.video:
        detect_vehicles_in_video(args.video)
    else:
        print("Por favor, forneeça um caminho para a imagem ou video usando --image ou --video")

if __name__ == '__main__':
    main()
