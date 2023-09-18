import cv2
import mediapipe as mp
from mediapipe.tasks import python
import threading 

class GestureRecognizer:

    is_new_hand = True

    hand_count_key = "hand_count"
    thumb_up_count_key = "thumb_up_count"
    thumb_down_count_key = "thumb_down_count"
    open_palm_count_key = "open_palm_count"

    count_data = {
        hand_count_key: 0,
        open_palm_count_key: 0,
        thumb_up_count_key: 0,
        thumb_down_count_key: 0
    }

    def main(self):
        num_hands = 1
        model_path = "models/gesture_recognizer.task"
        GestureRecognizer = mp.tasks.vision.GestureRecognizer
        GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        self.lock = threading.Lock()
        self.current_gestures = []

        options = GestureRecognizerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            num_hands = num_hands,
            result_callback=self.__result_callback)
        
        recognizer = GestureRecognizer.create_from_options(options)

        timestamp = 0 
        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=num_hands,
                min_detection_confidence=0.65,
                min_tracking_confidence=0.65)

        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            np_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # drawing landmarks
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # setting up image and recognizing expression
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_array)
                    recognizer.recognize_async(mp_image, timestamp)
                    timestamp = timestamp + 1 # should be monotonically increasing, because in LIVE_STREAM mode
                    
                self.put_gestures(frame)

            # Return image bytes
            ret, buffer = cv2.imencode('.jpg', frame)
            byte_buffer =  buffer.tobytes()

            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + byte_buffer + b'\r\n')

            # Display the image.
            # cv2.imshow('Hand Detection', frame)
            # if cv2.waitKey(1) & 0xFF == 27:
            #     break

        cap.release()

    def put_gestures(self, frame):
        self.lock.acquire()
        gestures = self.current_gestures
        self.lock.release()
        y_pos = 50
        blue = (107,41,0)
        yellow = (0,197,253)
        for hand_gesture_name in gestures:
            # show the prediction on the frame
            cv2.putText(frame, hand_gesture_name, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, blue, 6, cv2.LINE_AA)
            cv2.putText(frame, hand_gesture_name, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, yellow, 2, cv2.LINE_AA)
            y_pos += 50

    def __result_callback(self, result, output_image, timestamp_ms):
        #print(f'gesture recognition result: {result}')
        self.lock.acquire() # solves potential concurrency issues
        self.is_new_hand = not any(self.current_gestures)
        self.current_gestures = []
        if result is not None and any(result.gestures):
            print("Recognized gestures:")
            for single_hand_gesture_data in result.gestures:
                gesture_name = single_hand_gesture_data[0].category_name
                if self.is_new_hand:
                    match gesture_name:
                        case "Open_Palm":
                            self.count_data[self.open_palm_count_key] += 1
                            self.count_data[self.hand_count_key] += 1
                        case "Thumb_Up":
                            self.count_data[self.thumb_up_count_key] += 1
                            self.count_data[self.hand_count_key] += 1
                        case "Thumb_Down":
                            self.count_data[self.thumb_down_count_key] += 1
                            self.count_data[self.hand_count_key] += 1
                print(gesture_name)
                self.current_gestures.append(gesture_name)

        data = self.count_data
        print("Is new hand detected : " + str(self.is_new_hand))
        print("Hand Count : " + str(data[self.hand_count_key]))
        print("Open Palm Count : " + str(data[self.open_palm_count_key]))
        print("Thumbs Up Count : " + str(data[self.thumb_up_count_key]))
        print("Thumbs Down Count : " + str(data[self.thumb_down_count_key]))

        self.lock.release()

if __name__ == "__main__":
    rec = GestureRecognizer()
    rec.main()