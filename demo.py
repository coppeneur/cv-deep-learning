import argparse
import torch
import cv2
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from src import models as m


def preprocess_image(image, transform):
    """
    Helper function to preprocess a given image using the given transforms.
    :param image: image to preprocess
    :param transform: transform to apply
    :return: transformed image
    """
    image = Image.fromarray(image)
    image = transform(image).unsqueeze(0)
    return image


def get_emotion_prediction(image, model, device, transform):
    """
    Get the emotion probabilities for the given image using the given model.
    :param image: image to predict the emotion for
    :param model: model to use for prediction
    :param device: device to use for prediction
    :param transform: transform to apply to the image
    :return: probabilities
    """
    image = preprocess_image(image, transform)
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1).squeeze().cpu().numpy()
    return probabilities


def main(video_path=None):
    """
    Main function to run the emotion detection using the given video path or live camera.
    It will display the video feed with the emotion predictions and a summary plot at the end.
    The model used is the best model from the intermediateCNN model.
    :param video_path: optional video path to use for emotion detection if not provided, the live camera will be used
    :return:
    """
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    yellow_color = (0, 255, 255)
    red_color = (0, 0, 255)
    emotion_counts = Counter()

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    model = m.load_model(m.SimpleCNN(), 'bestmodels/Final_IntermediateCNN_CrossEntropyLoss_Adam_best_model.pth')
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Open the video file or webcam
    if video_path:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video file.")
            return
    else:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))

        # Draw rectangles around the faces and predict the emotion
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), yellow_color, thickness=7)
            roi_gray = gray_img[y:y + h, x:x + w]
            probabilities = get_emotion_prediction(roi_gray, model, device, transform)

            max_idx = np.argmax(probabilities)
            emotion_counts[emotion_labels[max_idx]] += 1

            for i, (emotion, prob) in enumerate(zip(emotion_labels, probabilities)):
                text = f"{emotion}: {prob:.2f}"
                if i == max_idx:
                    cv2.putText(frame, text, (x, y - 10 - i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, red_color, 2)
                else:
                    cv2.putText(frame, text, (x, y - 10 - i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, yellow_color, 1)

        cv2.imshow('Emotion Detection', frame)

        # Press aby jey to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    emotions, counts = zip(*emotion_counts.items())

    print(f'Total number of frames: {sum(counts)}')

    plt.figure(figsize=(10, 5))
    plt.bar(emotions, counts, color='skyblue')
    plt.xlabel('Emotions')
    plt.ylabel('Frames')
    plt.title('Emotion Detection Summary')
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Emotion detection using video input or live camera.')
    parser.add_argument('--video', type=str,
                        help='Path to the video file. If not provided, the live camera will be used.')
    args = parser.parse_args()

    main(video_path=args.video)
