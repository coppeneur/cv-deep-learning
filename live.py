import torch
import cv2
from torchvision import transforms
from PIL import Image
import numpy as np
from src import models as m
def preprocess_image(image):
    image = Image.fromarray(image)
    image = transform(image).unsqueeze(0)
    return image

def get_emotion_prediction(image):
    image = preprocess_image(image)
    image = image.to('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1).squeeze().cpu().numpy()
    return probabilities


## write a main function to run the code
if __name__ == '__main__':
    emotion_dict = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    # use saved model
    criterion = torch.nn.CrossEntropyLoss()
    model = m.load_model(m.SimpleCNN(), 'bestmodels/SimpleCNN_best_model_30_epochs.pth')
    model.eval()
    model.to('cuda' if torch.cuda.is_available() else 'cpu')

    cap = cv2.VideoCapture(0)
    # Assuming you have the emotion labels in a list
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    yellow_color = (0, 255, 255)
    red_color = (0, 0, 255)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame (you might want to use a better face detector)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), yellow_color, thickness=7)
            roi_gray = gray_img[y:y + h, x:x + w]
            probabilities = get_emotion_prediction(roi_gray)

            max_idx = np.argmax(probabilities)

            # Display all emotion confidences and highlight the one with the highest probability
            for i, (emotion, prob) in enumerate(zip(emotion_labels, probabilities)):
                text = f"{emotion}: {prob:.2f}"
                if i == max_idx:
                    cv2.putText(frame, text, (x, y - 10 - i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, red_color, 2)
                else:
                    cv2.putText(frame, text, (x, y - 10 - i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, yellow_color, 1)


        cv2.imshow('Live Emotion Detection', frame)

        if cv2.waitKey(1) != -1:
            break

    cap.release()
    cv2.destroyAllWindows()