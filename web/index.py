from flask import Flask, render_template, request, redirect, url_for, make_response,jsonify
from werkzeug.utils import secure_filename
from keras.layers import Input
import os
import cv2
import time
import sys
import datetime
import random
import keras.backend as K
from gevent.pywsgi import WSGIServer
import numpy as np
 
from datetime import timedelta
sys.path.append("..") 
from generator_conv import myGenerator


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def create_uuid(): 
    nowTime = datetime.datetime.now().strftime("%Y%m%d%H%M%S");  
    randomNum = random.randint(0, 100); 
    if randomNum <= 10:
        randomNum = str(0) + str(randomNum)
    uniqueNum = str(nowTime) + str(randomNum)
    return uniqueNum
 
app = Flask(__name__)
app.send_file_max_age_default = timedelta(seconds=1)
 
basepath = os.path.dirname(__file__)  


@app.route('/upload', methods=['POST', 'GET'])  
def upload():
    if request.method == 'POST':
        #K.clear_session()
        f = request.files['file']
 
        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": 'Incorrect file type'})
 
        user_input = request.form.get("name") 

        filename = create_uuid() + '.jpg'
        
        # filename =  secure_filename(f.filename)
        upload_path = os.path.join(basepath, 'static/images', filename) 
        f.save(upload_path)
        img = cv2.imread(upload_path)

        resultpath = do_predict(upload_path)
        #width, height = img.shape[:2]
        #resized = cv2.resize(img, (int(width * 0.3), int(height * 0.3)), cv2.INTER_CUBIC)
        #cv2.imwrite(os.path.join(basepath, 'static/images', 'test.jpg'), img)
        if resultpath == None:
            return render_template('upload.html', warning='Error occured when processing the file')
 
        return render_template('upload_ok.html', val1=time.time())
 
    return render_template('upload.html', warning='')



def do_predict(filepath):
    global generator
    img = cv2.imread(filepath)
    resized = cv2.resize(img, (224, 224), cv2.INTER_CUBIC)
    resized = np.array(resized, dtype=np.float)
    resized /= 255.0
    resized[84:140, 84:140] = np.zeros(shape=(56, 56, 3))
    print(resized.shape)
    result = generator.predict(resized.reshape(1, 224, 224, 3))
    resized[84:140, 84:140] = result
    print("prediction made")
    resultpath = os.path.join(basepath, 'static/images', 'processed.jpg')
    resized *= 255
    cv2.imwrite(resultpath, resized)
    return resultpath


@app.route('/download/<string:filename>', methods=['GET'])
def download(filename):
    if request.method == "GET":
        dirname = os.path.join(basepath, 'static/images')
        if os.path.isfile(os.path.join(dirname, filename)):
            return send_from_directory(dirname, filename, as_attachment=True)

@app.route('/show/<string:filename>', methods=['GET'])
def show_photo(filename):
    dirname = os.path.join(basepath, 'static/images', filename)
    if request.method == 'GET':
        if dirname is None:
            pass
        else:
            image_data = open(dirname, "rb").read()
            response = make_response(image_data)
            response.headers['Content-Type'] = 'image/png'
            return response
    else:
        pass

 
if __name__ == '__main__':
    global generator
    generator, _ = myGenerator(Input(shape=(224, 224, 3)))
    if os.path.exists("../generator.h5"):
        generator.load_weights("../generator.h5")
        # do test
        zeromat = np.zeros(shape=(1, 224, 224, 3))
        generator.predict(zeromat)
        # test over
        app.run(host='localhost', port=8080, debug=True)
    else:
        print("Weights not found!")

