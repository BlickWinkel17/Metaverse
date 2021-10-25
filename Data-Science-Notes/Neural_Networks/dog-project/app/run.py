import os
import urllib
from io import BytesIO
from PIL import Image

from flask import Flask
from flask import render_template, request, jsonify, redirect, url_for

import cv2         
import numpy as np
from keras.utils import np_utils
from keras.preprocessing import image 
# from keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint 
# from tqdm import tqdm

ALLOWED_EXTENSIONS = set(['png','PNG', 'jpg','JPG'])

# load features from Resnet50
bottleneck_features = np.load('model/bottleneck_features/DogResnet50Data.npz')
train_Resnet50 = bottleneck_features['train']
# valid_Resnet50 = bottleneck_features['valid']
# test_Resnet50 = bottleneck_features['test']

# load face detector
face_cascade = cv2.CascadeClassifier('model/haarcascades/haarcascade_frontalface_alt.xml')

# load ResNet50 models with and without the top layer
ResNet50_full = ResNet50(weights='imagenet')
ResNet50_cut = ResNet50(weights='imagenet', include_top=False)

# load dog_names.txt
dog_names = [dog_name.strip() for dog_name in open('model/dog_names.txt').readlines()]

# build model

Resnet50_model = Sequential()
Resnet50_model.add(GlobalAveragePooling2D(input_shape=train_Resnet50.shape[1:]))
Resnet50_model.add(Dense(133, activation='softmax'))

Resnet50_model.summary()
Resnet50_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# load the trained best model
Resnet50_model.load_weights('model/saved_models/weights.best.Resnet50.hdf5')


app = Flask(__name__)

# path_to_tensor
def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

# url_to_img:
def url_to_img(img_url):
    with urllib.request.urlopen(img_url) as url:
        img = Image.open(BytesIO(url.read()))
        img = img.convert('RGB')
        img = img.resize((224, 224), Image.NEAREST)    
    return img

# face detector
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

# dog detector
def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_full.predict(img))

def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 

# dog breed classifier

def Resnet50_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = ResNet50_cut.predict(preprocess_input(path_to_tensor(img_path)))
    # obtain predicted vector
    predicted_vector = Resnet50_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]

def Predict_breed(img_path):
    if face_detector(img_path):
        #print('=== Human detected, output the resembling dog breed ===')
        dog_name = Resnet50_predict_breed(img_path).split('.')[-1]
        return '=== Human detected, output the resembling dog breed === \n' + dog_name
    elif dog_detector(img_path):
        #print('=== Dog detected, output the predicted breed ===')
        dog_name = Resnet50_predict_breed(img_path).split('.')[-1]
        return '=== Dog detected, output the predicted breed === \n' + dog_name
    else:
        # print("=== Neither human nor dog is provided, please input a path of a human/dog image === \n")
        return "=== Neither human nor dog is provided, please input a path of a human/dog image === \n"

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    return render_template('master.html')


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    url = request.args.get('query', '') 

    if url == '' or url.split('.')[-1] not in ALLOWED_EXTENSIONS:
        return jsonify({"error":1001, "msg":"Please check your url"})

    else:
        try:
            url_img = url_to_img(url)
            img_name = url.split('/')[-1]
            basepath = os.path.dirname(__file__)
            save_path = basepath + '/static/images/' + img_name
            url_img.save(save_path)         

            print("img saved in {}".format(save_path))
            pred = Predict_breed(save_path.replace("\\","/"))

            # This will render the go.html Please see that file. 
            return render_template(
                'go.html',
                query = pred,
                filename = img_name
            )  

        except Exception as err:
            print(err)
            return render_template(
                'go.html',
                query = 'failed in analyzing the image url'
            )             
          

@app.route('/go', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']

    if uploaded_file.filename == '' or uploaded_file.filename.split('.')[-1] not in ALLOWED_EXTENSIONS:
        return jsonify({"error":1001, "msg":"Please check the format of your file"})
    else:
        basepath = os.path.dirname(__file__)
        save_path = basepath + '/static/images/' + uploaded_file.filename
        uploaded_file.save(save_path)
        print("img saved in {}".format(save_path))
        pred = Predict_breed(save_path.replace("\\","/"))

        # This will render the go.html Please see that file. 
        return render_template(
            'go.html',
            query = pred,
            filename = uploaded_file.filename
        )         

@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='images/' + filename), code=301)

def main():
    app.run(host='127.0.0.1', port=3001, debug=True)


if __name__ == '__main__':
    main()