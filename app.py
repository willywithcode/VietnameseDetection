from flask import Flask, render_template, Response,request
import cv2
import time
import imutils
import argparse
import f_Face_info  
import numpy as np
import os

app = Flask(__name__, static_folder = 'static')

@app.route('/')
def index():
    return render_template('index.html') 
@app.route('/camera_mode')
def camera():
    return render_template('index_cam.html')
@app.route('/image_mode', methods=['GET', 'POST'])
def image():
    if request.method == 'POST':
        if 'imageFile' in request.files:
            image = request.files['imageFile'].read()

            image_np = np.frombuffer(image, dtype=np.uint8)
            image_cv2 = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            out = f_Face_info.get_face_info(image_cv2)
            res_img = f_Face_info.bounding_box(out, image_cv2)
            # _, buffer = cv2.imencode('.jpg', res_img)
            # image_data = buffer.tobytes()
            image_filename = 'processed_image.jpg'  
            image_path = os.path.join(app.root_path, 'static', image_filename)
            cv2.imwrite(image_path, res_img)
            return render_template('index_image.html', image_path='/static/processed_image.jpg')
            # return render_template('index_image.html', image_data=image_data)
    return render_template('index_image.html')
def generate_frames():
    cam = cv2.VideoCapture(0)
    while True:
        star_time = time.time()
        ret, frame = cam.read()
        frame = imutils.resize(frame, width=720)

        out = f_Face_info.get_face_info(frame)
        res_img = f_Face_info.bounding_box(out, frame)
        end_time = time.time() - star_time    
        FPS = 1/end_time
        cv2.putText(res_img,f"FPS: {round(FPS,3)}",(10,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
        ret, buffer = cv2.imencode('.jpg', res_img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
