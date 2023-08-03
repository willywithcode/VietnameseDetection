from flask import Flask, render_template, Response
import cv2
import time
import imutils
import argparse
import f_Face_info  

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html') 

def generate_frames():
    cam = cv2.VideoCapture(0)
    while True:
        ret, frame = cam.read()
        frame = imutils.resize(frame, width=720)

        out = f_Face_info.get_face_info(frame)
        res_img = f_Face_info.bounding_box(out, frame)

        ret, buffer = cv2.imencode('.jpg', res_img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
