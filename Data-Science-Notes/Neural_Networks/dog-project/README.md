# Dog Breed Classifier Project

## Table of Contents
1. [Project Overview](#project-overview)
2. [Project Components](#project-components)
3. [File Description](#file-description)
4. [Installation](#installation)
5. [Discussions](#discussions)
6. [Screenshots of the web app](#screenshots)

## Project Overview
This project is part of the [Udacity Data Science Nano Degree Program](https://www.udacity.com/course/data-scientist-nanodegree--nd025). This project will analyze a data set containing dog images of different breeds. This project will build a dog breed classifier based on convolutional neural networks. A Flask web app is also included so that any one can input a image url or a local image and get a classification result.

## Project Components
There are two components in the project.

### 1. Dog breed classifier
- Build a Dog breed classifier based on ResNet50 by transfer learning. I removed its top layer, add a dense layer, fitted into our dataset and saved the best model. 
(This process has been done previously and model weight is saved under __\model\saved_models__. How the model was built is presented in __\model\dog_app.ipynb__)
- Loads the features and weights from the trained dog breed classifier.
- Once a local image or an image url is inputted via the web app, the related image is saved in __\app\templates\img__
(Currently __only *.png and *.jpg__ images are supported).
- The classifier takes the image path as input:
  - if a dog is detected, then the corresponding breed is returned.
  - if a human is detected, then the most resembling dog breed is returned.
  - if neither a dog nor a human is detected, then a notice is returned.
- (Optional) As you will see in __\model\saved_models__ and in the notebook, multiple models are provided (ResNet50 with image augmentation, VGG16, VGG19, InceptionV3). You are welcomed to try different models. However, since my current model is based on ResNet50, it's easy to try ResNet50 with image augmentation. All you have to do is load __weights.best.Resnet50_aug.hdf5 instead__ of __weights.best.Resnet50.hdf5__ in run.py. ResNet50 with image augmentation should provide slightly better classification results.

### 2. Flask Web App
There is a web app where any one can either upload a local image or input an image url, then a classification result will be presented (Examples screenshots can be found below). Notice that uploading a local image is recommended since getting an image from a url takes longer time and can even fail in some cases.

## File Description
```sh
- README.md: read me file
- \app
	- run.py: flask file to run the app. Model initialization also included.
   	- \templates
		  - master.html: main page of the web application 
		  - go.html: result web page
    - \static
      - \images 
        - images collected from web app 
- \model
  - \bottleneck_features
    - DogResnet50Data.npz: Resnet50 features
  - \haarcasrcades
    - haarcascade_frontalface_alt.xml: OpenCV pre-trained face detector
  - \images
    - some dog or human images to test with
  - \saved_models
    - weights.best.InceptionV3.hdf5: pre-trained InceptionV3 weights
    - weights.best.Resnet50_aug.hdf5: pre-trained Resnet50 weights with image augmentation
    - weights.best.Resnet50.hdf5: pre-trained Resnet50 weights
    - weights.best.VGG16.hdf5: pre-trained VGG16 weights
    - weights.best.VGG19.hdf5: pre-trained VGG19 weights
	- dog_app.ipynb: CNN model notebook
	- dog_names.txt: store names of 133 dog breeds
  - extract_bottleneck_features.py: different CNNs with top layer excluded
- \Screenshots
  - ClassifyViaUploadedImage.png
  - ClassifyViaImageUrl.png
```

## Installation
### Devendencies :
   - [python (>=3.6)](https://www.python.org/downloads/)  
   - [urllib](https://docs.python.org/3/library/urllib.html)
   - [os](https://docs.python.org/3/library/os.html)
   - [io](https://docs.python.org/3/library/io.html)
   - [cv2](https://pypi.org/project/opencv-python/)
   - [keras](https://keras.io/)
   - [tensorflow](https://www.tensorflow.org/learn)
   - [glob](https://docs.python.org/3/library/glob.html)
   - [tqdm](https://github.com/tqdm/tqdm)
   - [pandas](https://pandas.pydata.org/)  
   - [numpy](https://numpy.org/)  
   - [sys](https://docs.python.org/3/library/sys.html)  
   - [sklearn](https://sklearn.org/)  
   - [flask](https://flask.palletsprojects.com/en/2.0.x/)  
   
 ### Download and Installation
 ```sh
git clone https://github.com/BlickWinkel17/DogBreedClassifier.git
```
Entering the project path
 ```sh
cd DogBreedClassifier
```
Next, run the Python file run.py.
```sh
python app/run.py
```
Finally, go to http://127.0.0.1:3001/ in your web-browser.
Upload a local image directly or type an image url to see the dog breed that your image falls into.

## Discussions

- This ResNet50 based model achieved an accuracy of around __80%__ on a test set of 836 dog images. Other models such as VGG16, VGG19 and InceptionV3 were also tested, none of which exceeds that accuracy. But most satisfying records found in Kaggle reported accuracies over 90%, indicating the fact that the current model can be improved further much more.
- In building the model, I tried an simple image augmentation approach(see the Image Augmentation part in the notebook), which slightly improves the result. This image augmentation can be investigated deeplier.
- If we look at these wrong predictions in test set, it turns out that the current model fails to classify correctly in some resembling dog breeds. For example, the current model tends to recognize __German wirehaired pointer__ as __Wirehaired pointing griffon__, and recognize __German pinscher__ as __Doberman pinscher__. It seems that the model fails to capture some slight differences between dog breeds.

<div align=center>

__German wirehaired pointer and Wirehaired pointing griffon__

<img src="https://github.com/BlickWinkel17/DogBreedClassifier/blob/master/Screenshots/German%20wirehaired%20pointer.jpg" width="300" height="300"> <img src="https://github.com/BlickWinkel17/DogBreedClassifier/blob/master/Screenshots/Wirehaired%20pointing%20griffon.jpg" width="300" height="300"> 
<div>
	
<div align=center>

__German pinscher and Doberman pinscher__

<img src="https://github.com/BlickWinkel17/DogBreedClassifier/blob/master/Screenshots/German%20pinscher.jpg" width="300" height="300"> <img src="https://github.com/BlickWinkel17/DogBreedClassifier/blob/master/Screenshots/Doberman%20pinscher.jpg" width="300" height="300">
<div>

## Screenshots
	
![Alt text](https://github.com/BlickWinkel17/DogBreedClassifier/blob/master/Screenshots/ClassifyViaUploadedImage.png)
![Alt text](https://github.com/BlickWinkel17/DogBreedClassifier/blob/master/Screenshots/ClassifyViaImageUrl.png)
