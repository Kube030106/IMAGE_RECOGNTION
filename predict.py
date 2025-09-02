import cv2
import joblib
import numpy as np

# Load the trained model
model=joblib.load("knn_model.pkl")

#load the testt image
test_image_path="elep2.jpeg"
img=cv2.imread(test_image_path,cv2.IMREAD_GRAYSCALE)

#resize
img=cv2.resize(img,(100,100))
img_flat=img.flatten().reshape(1,-1)#reshape(1,10000)
#Predict
prediction=model.predict(img_flat)
print("Prediction: ",prediction[0])