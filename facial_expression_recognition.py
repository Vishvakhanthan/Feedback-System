import cv2
from keras.models import model_from_json
import numpy as np
import mediapipe as mp

# face detection model
face_detector = mp.solutions.face_detection.FaceDetection()

# Facial Expression Detection Model
json_file = open("models/facial_expression_recognition.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("models/facial_expression_recognition.h5")

# extracting features from image and normalize it
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# labels for respective expressions
labels = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise",
}

# Process an image and detect faces.
cap = cv2.VideoCapture(0)
blue = (107,41,0)
yellow = (0,197,253)


while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break

    height, width, _ = frame.shape

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Draw the detected faces on the image.
    if results.detections:
        for face_detection in results.detections:
            bounding_box = face_detection.location_data.relative_bounding_box
            x, y, w, h = (
                int(bounding_box.xmin * width),
                int(bounding_box.ymin * height),
                int(bounding_box.width * width),
                int(bounding_box.height * height),
            )
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2,
            )

            # extract face from frame
            detected_face = frame[y:y+h, x:x+w]
            detected_face = cv2.cvtColor(detected_face, cv2.COLOR_RGB2GRAY)
            detected_face = cv2.resize(detected_face, (48, 48))

            # predict expression
            img = extract_features(detected_face)
            expression = model.predict(img)
            expression_label = labels[expression.argmax()]

            # printing expression label
            cv2.putText(frame, expression_label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, blue, 6, cv2.LINE_AA)
            cv2.putText(frame, expression_label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, yellow, 2, cv2.LINE_AA)

            # first_frame = True
            # if first_frame:
            #     cv2.imwrite(f"face.jpg", detected_face)


    # Display the image.
    cv2.imshow("Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
