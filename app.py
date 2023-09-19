from flask import Flask, render_template, Response

from gesture_recognition import GestureRecognizer
from expression_recognition import ExpressionRecognizer

gesture_recognizer = GestureRecognizer()
expression_recognizer = ExpressionRecognizer()


app = Flask(__name__, static_folder="templates")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/hand_gesture')
def hand_gesture():
    return Response(gesture_recognizer.main(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/gesture_count')
def gesture_count():
    (good, average, bad) = gesture_recognizer.gesture_count()
    return f"{good}, {average}, {bad}"

@app.route('/face_expression')
def face_expression():
    return Response(expression_recognizer.main(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
