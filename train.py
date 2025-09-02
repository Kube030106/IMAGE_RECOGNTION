import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib

#folder where images
data_dir='dataset'
# List to store data 

X=[]
y=[]
#resize all image

img_size=(100,100)

# Go through each folder
for label in os.listdir(data_dir):
    class_folder=os.path.join(data_dir,label)

    #Go through the images
    for file in os.listdir(class_folder):
        img_path=os.path.join(class_folder,file)

        #load img in grey scale
        img=cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)

        #resize the image
        img=cv2.resize(img,img_size)

        features=img.flatten()

        #Store features and label
        X.append(features)
        y.append(label)
#convert the list to numpy arrays
X=np.array(X)
y=np.array(y)

#create KNN model
model=KNeighborsClassifier(n_neighbors=1)

#train model
model.fit(X,y)
joblib.dump(model,"knn_model.pkl")
print("Model training completed")
print("Total samples: ",len(X))
print("Labels: ",np.unique(y))