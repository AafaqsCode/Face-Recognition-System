#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 13:20:40 2020

@author: Aafaq
"""
from keras.preprocessing.image import img_to_array, array_to_img, img_to_array, load_img
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
from keras import backend as K
import glob
import random  as rnd
import numpy as np
import pandas as pd
import operator
import random 
import seaborn as sns

# size of images 
IMG_HEIGHT = 92
IMG_WIDTH = 112

# let load the 2nd image in the folder s1


n_fldr = 1
n_img = 2
img = load_img('orl_faces'+ '/s' + str(n_fldr) + '/' + str(n_img) + '.pgm',
               target_size=(IMG_WIDTH, IMG_HEIGHT))
# show image
img.show()


# display some random images from random folder

fig = plt.figure(figsize=(8, 6))
# plot several images
for i in range(15):
    rand_fldr = rnd.randint(1,40)  # random folder 
    rand_img = rnd.randint(1,10)   # random images for that folder
    ax = fig.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
    path_img = 'orl_faces/s' + str(rand_fldr) + '/' + str(rand_img) + '.pgm'
    img = load_img(path_img,target_size=(IMG_WIDTH, IMG_HEIGHT))
    plt.suptitle('random images')
    ax.imshow(img, cmap=plt.cm.bone)
    

'''
load image

'''
title = "Loading the images with size 48x48"
print(title.center(100, '='))

img = load_img('orl_faces/s1/1.pgm',target_size=(IMG_WIDTH, IMG_HEIGHT))
img_ar = np.array(img)
print(img_ar.shape)  # 3d (channel) images
'''
Assume we work with one channel image. Take one channel (2 dimension)
'''
img_2d = img_ar[:,:,0]
plt.imshow(img_2d)

'''
convert 2d image to 1d vector. Just by reshaping it
'''
title = "Converting images to 1d vector"
print(title.center(100, '='))

img_1d = img_2d.reshape(-1)
print(img_1d.shape)  #IMG_WIDTH * IMG_HEIGHT
'''
The size of the vector os 10304, which is large for processing by KNN.
So, it is preferable to load the image with smaller size. Let say 48*48
'''
print("Reduce images size".center(100, '='))
new_hgth = 48
new_wdth = 48
img = load_img('orl_faces/s1/1.pgm',target_size=(new_hgth, new_wdth))
img_ar = np.array(img)
img_2d = img_ar[:,:,0]
img_1d = img_2d.reshape(-1)
print(img_1d.shape)  #Inow it is 2304

# -------------------------

'''
3. Load them in one matrix named F of size 400x2304. ( number of images x number of features)
'''

folders = 40
images = 10
fig2 = plt.figure(figsize=(8, 6))
F = []
for i in range(1,folders+1):
 for j in range(1,images+1):
     n_fldr = i
     n_img = j
     img = load_img('orl_faces'+ '/s' + str(n_fldr) + '/' + str(n_img) + '.pgm', 
               target_size=(new_wdth, new_hgth))
     img_ar = np.array(img)
     img_2d = img_ar[:,:,0]
     img_1d = img_2d.reshape(-1)
     F.append(img_1d)

X = np.array(F)
     

print(np.shape(F))



'''
4. Create a vector y_true of size 400. y_true is the vector of true labels of the
images. It holds 10 labels 0, 10 lables 1,... 10 labels 40.
'''
y_true = []
for i in range(1,folders+1):
 for j in range(1,images+1):
     s= str(i) 
     y_true.append(s)


'''
5. Scale feature with different feature scaling (normalization).
(standarization) Eigenfaces, 
'''

title = "Normalization"
print(title.center(100, '='))

from sklearn import preprocessing

# 1. Standarization (z-score normalization) X_ss

std_scaler = preprocessing.StandardScaler().fit(X)
X_ss = std_scaler.transform(X)
# let compare mena and std of the first features of X and X_ss
print("compare mean and std of the first features of X and X_ss")
f1_X = X[:,0]
print('%.2f %.2f'%(np.std(f1_X), np.mean(f1_X)))

f1_Xss = X_ss[:,0]
print('%.2f %.2f'%(np.std(f1_Xss), np.mean(f1_Xss)))


# 2. Min max scaling X_mm

minmax_scaler = preprocessing.MinMaxScaler().fit(X)
X_mm = minmax_scaler.transform(X)
# let comprae min and max of first features of X and X_mm
print("compare min and max of first features of X and X_mm")
f1_X = X[:,0]
print('%.2f %.2f'%(np.min(f1_X), np.max(f1_X)))

f1_Xmm = X_mm[:,0]
print('%.2f %.2f'%(np.min(f1_Xmm), np.max(f1_Xmm)))

# 3. Binarizing X_Binarize

X_Binarize =preprocessing.Binarizer(0.0).fit(X).transform(X)
# let comprae Binarizing of first features of X and X_Binarize
print("compare Binarizing of first features of X and X_Binarize")
f1_X = X[0]
print(f1_X)

f1_X_Binarize = X_Binarize[0]
print(f1_X_Binarize)

# 3. Normalizing

X_Normalize = preprocessing.Normalizer().fit(X).transform(X)
# let comprae Binarizing of first features of X and X_Normalize
print("compare Normalizing of first features of X and X_Normalize")
f1_X = X[0]
print(f1_X)

f1_X_Normalize = X_Normalize[0]
print(f1_X_Normalize)


'''
Tried PCA!
'''

from glob import iglob
faces = pd.DataFrame([])
for path in iglob("orl_faces/*/*.pgm"):
 img=imread(path)
 face = pd.Series(img.flatten(),name=path)
 faces = faces.append(face)
 
fig, axes = plt.subplots(3,5,figsize=(8,6),
 subplot_kw={"xticks":[], "yticks":[]},
 gridspec_kw=dict(hspace=0.01, wspace=0.01))
plt.suptitle('images befoe appling PCA')
for i, ax in enumerate(axes.flat):
 ax.imshow(faces.iloc[i].values.reshape(112,92),cmap="gray")

#n_components=0.80 means it will return the Eigenvectors that have the 70% of the variation in the dataset
faces_pca = PCA(n_components=0.7)
faces_pca.fit(faces)
 
components = faces_pca.transform(faces)
projected = faces_pca.inverse_transform(components)
fig, axes = plt.subplots(3,5,figsize=(8,6), subplot_kw={'xticks':[], 'yticks':[]},
            gridspec_kw=dict(hspace=0.01, wspace=0.01))
plt.suptitle('Tried PCA!')
for i, ax in enumerate(axes.flat):
    ax.imshow(projected[i].reshape(112,92),cmap="gray")

# illustartion of how PCA works
import mglearn
mglearn.plots.plot_pca_illustration()
'''
6. Split dataset into training and test subsets.
70 percent will be used for training, 30 percent for testing
'''
x_train, x_test, y_train, y_test = [],[],[],[]
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)  



def split(x):
    X = x
    Y = np.array(y_true)
    from sklearn.model_selection import train_test_split
    global x_train
    global x_test
    global y_train
    global y_test
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

 
'''
7. Apply K-NN classifier.
'''


def getEuclidDist2Vectors(v1, v2):
    return np.sqrt(np.sum((v1-v2)**2))


def KNN(x_train, y_train, x_test, K):
 
    y_predicted = []
    distances = {}
    sz_train = x_train.shape[0]   # number of test samples in x_train
    
    # for each test sample.....
    for xtst in x_test:
        #... compute the distance between xtst it to each train sample
        for i in range(sz_train):
            dst = getEuclidDist2Vectors(xtst, x_train[i,:])
            distances[i] = dst
        
        # sort the distances from smalest to largest
        distSorted = sorted(distances.items(), key=operator.itemgetter(1))
    
     
        # get the k neares neighbors in a list <neighbors>
        neighbors = []
        for k in range(K):
            neighbors.append(distSorted[k][0])
            
        # get the labels of the first nearest neighbors from the x_train in list <lbl>
        lbl = []
        votes = {}
        
        for n in neighbors:
            l = y_train[n]
            lbl.append(y_train[n])
            if l in votes:
                votes[l] +=1
            else:
                votes[l] = 1
     
        votesSorted = sorted(votes.items(), key=operator.itemgetter(1), reverse=True)
        dominantLabel = votesSorted[0][0]
        
        y_predicted.append(dominantLabel)
        
    return y_predicted


'''
predict_KNN(k) method 
'''
from sklearn.metrics import accuracy_score

def predict_KNN(X_value):
    X = X_value
    split(X)
    plt.figure(1, figsize=(12,8))
    # if we want to get the prediction of each class
#    y_pred = KNN(x_train, y_train, x_test, k)   
#    acc_score = accuracy_score(y_test, y_pred)
#    print('the accuracy score for KNN with k= %d is %.2f'%(k,acc_score))
#    for i, j in zip(y_pred , y_test):
#        print( " the algo predict the class {}, while the real class is {}".format(i,j))
#    
#    -----------------------------
    error = []
    # Calculating error for K values between 1 and 40
    for i in range(1, 40):
        pred_i = KNN(x_train, y_train, x_test, i)
        error.append(np.mean(pred_i != y_test))
        
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
    plt.title('Error Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Mean Error')
'''
KNN performance with different values of K
'''

title = "KNN performance with different k values"
print(title.center(100, '='))

predict_KNN(X)


'''
get_feature_accuracy(X) method 
'''


def get_feature_accuracy(X):
    split(X)
    y_pred = KNN(x_train, y_train, x_test, 3)
    acc_score = accuracy_score(y_test, y_pred)
    # classification report
#    from sklearn import metrics
#    sns.heatmap(metrics.confusion_matrix(y_test, y_pred))
#    print("classification_report",classification_report(y_test, y_pred))
    return acc_score




def accuracy(y_test, y_pred):
    return K.mean(K.equal(y_test, K.cast(y_pred < 0.5, y_test.dtype)))


'''
KNN performance with different feature scaling techniques
'''


title = "KNN performance with different feature scaling techniques"
print(title.center(100, '='))

##----- KNN before feature scaling
print("KNN before feature scaling")
X__feature_accuracy = get_feature_accuracy(X)
print('%.2f'%(X__feature_accuracy))

##----- KNN after feature standard scaling
print("KNN after feature standard scaling")
X_ss_accuracy = get_feature_accuracy(X_ss)
print('%.2f'%(X_ss_accuracy))

##----- KNN after feature Min max scaling 
print("KNN after feature Min max scaling")
X_mm_accuracy = get_feature_accuracy(X_mm)
print('%.2f'%(X_mm_accuracy))

##----- KNN after Binarizing
print("KNN after Binarizing")
X_Binarize_accuracy = get_feature_accuracy(X_Binarize)
print('%.2f'%(X_Binarize_accuracy))

##----- KNN after Normalizing
print("KNN after Normalizing ")
X_Normalize_accuracy = get_feature_accuracy(X_Normalize)
print('%.2f'%(X_Normalize_accuracy))

# function to show the plot 
plt.show() 

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
titles = ['before', 'MinMax scaling', 'standard scaling', 'Normalizing', 'Binarizing'] 
features = [X__feature_accuracy, X_mm_accuracy, X_ss_accuracy, X_Normalize_accuracy, X_Binarize_accuracy] 
ax.bar(titles,features)
plt.show()




'''
KNN performance vs K for each feature scaling techniques
'''

from termcolor import colored
title = "KNN performance vs K for each feature scaling techniques"
print(title.center(100, '='))

x_values= [X,X_ss,X_mm,X_Binarize,X_Normalize]
colours = [ "red", "blue", "green", "yellow", "cyan","magenta"]
plt.figure(figsize=(12, 6))
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')

errors = []
    
 # Calculating error for K values between 1 and 40
for j in range(len(titles)):
    t = titles[j]
    X = x_values[j]
    split(X)
    error = []    
    for i in range(1, 40):
        pred_i = KNN(x_train, y_train, x_test, i)
        error.append(np.mean(pred_i != y_test))  
        errors.append(np.mean(pred_i != y_test))  
    
    
    color = random.choice(colours) 
    colours.remove(color)
    print(colored(t, color)) 
    plt.plot(range(1, 40), error, color=color, linestyle='dashed', marker='o',
                 markerfacecolor=color, markersize=10)
    
