import cv2
from keras.models import model_from_json
import numpy as np
import mediapipe as mp

class ExpressionRecognizer():

    # extracting features from image and normalize it
    def extract_features(self, image):
        feature = np.array(image)
        feature = feature.reshape(1, 48, 48, 1)
        return feature / 255.0

    def main(self):

        # face detection model
        face_detector = mp.solutions.face_detection.FaceDetection()

        # Facial Expression Detection Model
        json_file = open("models/facial_expression_recognition.json", "r")
        model_json = json_file.read()
        json_file.close()
        model = model_from_json(model_json)
        model.load_weights("models/facial_expression_recognition.h5")

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

        cap = cv2.VideoCapture(0)

        blue = (107,41,0)
        yellow = (0,197,253)


        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
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
                        (255, 255, 255),
                        2,
                    )

                    # extract face from frame
                    detected_face = frame[y:y+h, x:x+w]
                    detected_face = cv2.cvtColor(detected_face, cv2.COLOR_RGB2GRAY)
                    detected_face = cv2.resize(detected_face, (48, 48))

                    # predict expression
                    img = self.extract_features(detected_face)
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

            ret, buffer = cv2.imencode('.jpg', frame)
            byte_buffer =  buffer.tobytes()

            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + byte_buffer + b'\r\n')

            # Display the image.
            # cv2.imshow("Face Detection", frame)
            # if cv2.waitKey(1) & 0xFF == 27:
            #     break

        cap.release()

if __name__ == "__main__":
    rec = ExpressionRecognizer()
    rec.main()