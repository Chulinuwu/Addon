import cv2
import numpy as np
from keras.models import load_model
import tensorflow as tf
import time

# Load the pre-trained regression model
model = load_model('model/plant_disease_regression_model.h5')

# Load the MobileNet SSD model
net = cv2.dnn.readNetFromCaffe('model/deploy.prototxt', 'model/mobilenet_iter_73000.caffemodel')

def detect_people(frame):
    # Prepare the frame for detection
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    people_boxes = []

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])
            if idx == 15:  # Class label for person in COCO dataset
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")
                people_boxes.append((startX, startY, endX, endY))
    return people_boxes

def predict_disease(frame):
    # Resize and preprocess the image
    image_resized = cv2.resize(frame, (256, 256))
    image_array = np.array(image_resized)
    image_array = np.expand_dims(image_array, axis=0)

    # Predict the disease probability
    predictions = model.predict(image_array)
    probability = predictions[0][0]

    # Debugging information
    print(f'Predictions: {predictions}')
    print(f'Probability: {probability}')

    return probability

def main():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Detect people in the frame
        people_boxes = detect_people(frame)

        # Assuming the whole frame is a leaf for simplicity
        if not people_boxes:
            probability = predict_disease(frame)
            healthy_prob = 1 - probability
            diseased_prob = probability

            # Determine the label and color based on the probability
            if diseased_prob > 0:
                label = f'Diseased Probability is Higher than Healthy Probability : {diseased_prob:.2f} ' 
                color = (0, 0, 255)  # Red for diseased
            else:
                label = f'Healthy Probability is Higher than Diseased Probability :  {healthy_prob:.2f}'
                color = (0, 255, 0)  # Green for healthy

            # Draw rectangle around the leaf (whole frame in this case)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), color, 2)
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Display the results
            print(f'Healthy Probability: {healthy_prob}, Diseased Probability: {diseased_prob}')

        # Display the frame
        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()