# Baseline MLP for MNIST dataset
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, LSTM, GlobalAveragePooling1D
from keras.layers import Dropout
from keras.utils import np_utils
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle
import itertools
from sklearn.preprocessing import normalize

#import skimage.measure

# global variables
lstm=0         # flag for turning on/off LSTM model
time_window=0  # for no time-windowed data, set to 0
batch_size=1024 # time-window batch size

# function to map index with maximum value to respective predicted class
def array_to_num(x):
    x = list(x)
    res = [list(a).index(max(a)) for a in x]
    fit_dummy = {0: 'BL', 1: 'BR', 2:'CC', 3:'FL', 4:'FR'}        # one hot encoding [BL BR CC FL FR] => [0 2 3 4] => BR is index 0 while FR is index 4.
    pos = [fit_dummy[a] for a in res]
    return pos

# generate Receiver Operating Characteristic (ROC) Curve
def generate_ROC(y_test, y_score,n_classes):
   # Compute ROC curve and ROC area for each class
   fpr = dict()
   tpr = dict()
   roc_auc = dict()
   for i in range(n_classes):
       fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
       roc_auc[i] = auc(fpr[i], tpr[i])

   # Compute micro-average ROC curve and ROC area
   fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
   roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
   plt.figure()
   plt.plot(fpr[0], tpr[0], label='ROC curve (area = %0.2f)' % roc_auc[0])
   plt.plot([0, 1], [0, 1], 'k--')
   plt.xlim([0.0, 1.05])
   plt.ylim([0.0, 1.05])
   plt.xlabel('False Positive Rate')
   plt.ylabel('True Positive Rate')
   plt.title('Receiver operating characteristic curve')
   plt.show()
   print('AUC: %f' % roc_auc[0])
   return fpr, tpr, roc_auc
    
# Compute macro-average ROC curve and ROC area ---PENDING IMPLEMENTATION-----
# First aggregate all false positive rates
def generate_multiclass_ROC(fpr,tpr,roc_auc,n_classes):
   all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

   # Then interpolate all ROC curves at this points
   mean_tpr = np.zeros_like(all_fpr)
   for i in range(n_classes):
      mean_tpr += interp(all_fpr, fpr[i], tpr[i])

   # Finally average it and compute AUC
   mean_tpr /= n_classes
   lw = 2
   
   fpr["macro"] = all_fpr
   tpr["macro"] = mean_tpr
   roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

   # Plot all ROC curves
   plt.figure()
   ''' plot micro & macro ROC curves
   plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

   plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)
   '''
   colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
   for i, color in zip(range(n_classes), colors):
      plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

   plt.plot([0, 1], [0, 1], 'k--', lw=lw)
   plt.xlim([0.0, 1.0])
   plt.ylim([0.0, 1.05])
   plt.xlabel('False Positive Rate')
   plt.ylabel('True Positive Rate')
   plt.title('Multi-Class ROC')
   plt.legend(loc="lower right")
   plt.show()

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    cm=np.around(cm, decimals=3)  # round to 3 decimal precision after decimal
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



#-----> filter section <------
# ---->LOW PASS FILTERS<------
"""
OpenCV provides a function, cv2.filter2D(), to convolve a kernel with an image.
Filtering with the above kernel results in the following being performed:
for each pixel, a 5x5 window is centered on this pixel, all pixels falling
within this window are summed up, and the result is then divided by 25. This
equates to computing the average of the pixel values inside that window.
This operation is performed for all the pixels in the image to produce the
output filtered image. Try this code and check the result:
"""
def averaging_filter1(img):
    kernel = np.ones((2,2),np.float32)/4
    for i in range(img.shape[0]):
        img[i,:,:] = cv2.filter2D(img[i,:,:],-1,kernel)   # averaged image
    return img

"""
Averaging
This is done by convolving the image with a normalized box filter. It simply
takes the average of all the pixels under kernel area and replaces the central
element with this average. This is done by the function cv2.blur() or
cv2.boxFilter(). Check the docs for more details about the kernel. We should
specify the width and height of kernel. A 3x3 normalized box filter would
look like this
"""
def averaging_filter2(img):
   for i in range(img.shape[0]):
       img[i,:,:] = cv2.blur(img[i,:,:],(5,5))
   return img

"""
Gaussian Filtering
In this approach, instead of a box filter consisting of equal filter
coefficients, a Gaussian kernel is used. It is done with the function,
cv2.GaussianBlur(). We should specify the width and height of the kernel
which should be positive and odd. We also should specify the standard
deviation in the X and Y directions, sigmaX and sigmaY respectively. If only
sigmaX is specified, sigmaY is taken as equal to sigmaX. If both are given as
zeros, they are calculated from the kernel size. Gaussian filtering is highly
effective in removing Gaussian noise from the image.
"""
def gaussian(img):
   for i in range(img.shape[0]):
       img[i,:,:] = cv2.GaussianBlur(img[i,:,:],(5,5),0)
   
   return img

"""
Here, the function cv2.medianBlur() computes the median of all the pixels
 under the kernel window and the central pixel is replaced with this median
 value. This is highly effective in removing salt-and-pepper noise. One
 interesting thing to note is that, in the Gaussian and box filters, the
 filtered value for the central element can be a value which may not exist
 in the original image. However this is not the case in median filtering,
 since the central element is always replaced by some pixel value in the
 image. This reduces the noise effectively. The kernel size must be a
 positive odd integer.
"""
def bilateral(img):
    for i in range(img.shape[0]):
        img[i,:,:] = cv2.medianBlur(img[i,:,:],5)
    return img

"""
As we noted, the filters we presented earlier tend to blur edges. This is not
 the case for the bilateral filter, cv2.bilateralFilter(), which was defined
 for, and is highly effective at noise removal while preserving edges. But the
 operation is slower compared to other filters. We already saw that a Gaussian
 filter takes the a neighborhood around the pixel and finds its Gaussian
 weighted average. This Gaussian filter is a function of space alone, that is,
 nearby pixels are considered while filtering. It does not consider whether
 pixels have almost the same intensity value and does not consider whether the
 pixel lies on an edge or not. The resulting effect is that Gaussian filters
 tend to blur edges, which is undesirable.
"""
def median(img):
    for i in range(img.shape[0]):
        img[i,:,:] = cv2.bilateralFilter(img[i,:,:],9,75,75)
    return img

#-----> HIGH PASS FILTERS <------
"""
OpenCV provides three types of gradient filters or High-pass filters, Sobel,
Scharr and Laplacian.Sobel operators is a joint Gausssian smoothing plus
differentiation operation, so it is more resistant to noise. You can specify
the direction of derivatives to be taken, vertical or horizontal (by the
arguments, yorder and xorder respectively). You can also
specify the size of kernel by the argument ksize. If ksize = -1, a 3x3 Scharr
filter is used which gives better results than 3x3 Sobel filter. Please see
the docs for kernels used.

"""
def laplacian(img):
    for i in range(img.shape[0]):
        img[i,:,:] = cv2.Laplacian(img[i,:,:],cv2.CV_64F)
    return img

def sobelx(img):
    for i in range(img.shape[0]):
        img[i,:,:] = cv2.Sobel(img[i,:,:],cv2.CV_64F,1,0,ksize=5)  # x
    return img

def sobely(img):
    for i in range(img.shape[0]):
        img[i,:,:] = cv2.Sobel(img[i,:,:],cv2.CV_64F,0,1,ksize=5)  #y
    return img

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()


'''
# THIS IDEA OF CASCADING FILTERS BEFORE FFT IN NOT WORKING....
filters = [averaging_filter1,averaging_filter2,bilateral,median, gaussian, laplacian, sobelx, sobely]
train = [filt(X_train) for filt in filters]
test = [filt(X_test) for filt in filters]

fft_train = [np.fft.fft2(x) for x in train]
fft_test = [np.fft.fft2(x) for x in test]

fft_shift_train = [np.fft.fftshift(x,axes=(1,2)) for x in fft_train]
fft_shift_test = [np.fft.fftshift(x,axes=(1,2)) for x in fft_test]

X_train = np.vstack(fft_shift_train)
X_test = np.vstack(fft_shift_test)

y_train = np.vstack(len(filters)*[y_train])
y_test = np.vstack(len(filters)*[y_test])
#-------------------------------------->
'''

# graceful exit from Spyder
#raise Exception('exit')

"""
NOW THAT THE DATA IS IN FOURIER SPACE, DO SOME SIGNAL PROCESSING TO ELIMINATE
REDUDANT INFORMATION IN THE DATA WHCIH MEANS USING ALL THE SIGNAL PROCESSING TOOLS
TO PROCESS THE DATA PRIOR TO INPUT INTO A NEURON NETWORK
1) use only phase image for neuron network

#print("X_train:",X_train.shape)
"""
#---> convert to spectral domain and process neuron network using FFT2 data<--

X_train=np.fft.fft2(X_train)         # train data
X_test=np.fft.fft2(X_test)           # test data

# fftshift over the image axes (x,y) only, default is all !
X_train=np.fft.fftshift(X_train, axes=(1,2))
X_test=np.fft.fftshift(X_test, axes=(1,2))


""" IDEAS TO INVESTIGATE...
X_train=np.fft.fftshift(X_train, axes=(1,2))
2) use only magnitude image for neuron network
3) use magnitude spectrum
3) Filter out unnecessary information to reduce feature representation
4) real component only
5) imaginary compoonent only
6) Look at pooling that is used with 1D convolutions
7) batch normalization
8) object recognition in photographs CIFAR-10 data set Ch 21
9) a new dropout/pooling scheme of skipping points in the fourier image
10) create a random matrix of zeros & ones and use it as a mask on the training
    images as a form of pooling. Need element by element matrix multiplication
"""

'''
# does not work
drop=np.random.choice([1, 0], size=(X_train.shape[1],X_train.shape[2]), p=[0.8, 0.2])
for i in range(X_train.shape[0]):
    X_train.real[i,:,:]=X_train.real[i,:,:]*drop
    X_train.imag[i,:,:]=X_train.imag[i,:,:]*drop
'''    

#'''
mag=0            # mag=1: use mag images only
phase=0          # phase=1: use phase images only
if mag==1:   # use phase or magnitude image
   X_train=abs(X_train)
   X_test=abs(X_test)
elif phase==1:        # use phase images only
    X_train=np.angle(X_train)
    X_test=np.angle(X_test)
else:                # combine real & imag images
   # combine data =[real imag] for training/testing
   print(X_train.shape)
   X_train=np.concatenate((X_train.real,X_train.imag),axis=1)
   X_test=np.concatenate((X_test.real,X_test.imag),axis=1)
   

# graceful exit from Spyder
#raise Exception('exit')
#'''

'''
# numpy based max pooling of the training data is applied here
X_pool_train=np.zeros((X_train.shape[0],X_train.shape[1]//2,X_train.shape[2]//2))  # integer division
X_pool_test=np.zeros((X_test.shape[0],X_test.shape[1]//2,X_test.shape[2]//2))
for i in range(X_train.shape[0]):
    X_pool_train[i,:,:]=skimage.measure.block_reduce(X_train[i,:,:], (2,2), np.min)
for i in range(X_test.shape[0]):    
    X_pool_test[i,:,:]=skimage.measure.block_reduce(X_test[i,:,:], (2,2), np.min)
X_train=X_pool_train
X_test=X_pool_test
'''

# Novel Method to randomly drop pixels in the Fourier Space - new form of dropout
# YOu NEED TO RUN THIS IN A BATCH TYPE SETUP AND USE THIS MODULE TO RANDOMLY DROP PIXELS
# DURING EACH BATCH TRAINING...
# ----> IT IS NOT WORKING <-----
'''
# does not work
drop=np.random.choice([1, 0], size=(X_train.shape[1],X_train.shape[2]), p=[0.8, 0.2])
for i in range(X_train.shape[0]):
    X_train[i,:,:]=X_train[i,:,:]*drop
'''    

'''    
drop=np.random.choice([0, 1], size=(X_test.shape[1],X_test.shape[2]), p=[1./3, 2./3])
for i in range(X_test.shape[0]):
    #X_test[i,:,:]=X_test[i,:,:]*drop[0:X_test.shape[0],0:X_test.shape[1]]
    X_test[i,:,:]=X_test[i,:,:]*drop
'''
   
print(X_train.shape)
print(X_test.shape)
#X_train=X_train[:,::2]
#X_test=X_test[:,::2]

# flatten 28*28 images to a 784 vector for each image
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2]).astype('float32')
print("X_train2:", X_train.shape)

# normalize inputs from 0-255 to 0-1
X_train = X_train / max(X_train.flatten()) #255
X_test = X_test / max(X_test.flatten())

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

if lstm==1:
   # reshape input to be [samples, time steps, features]
   X_train=np.reshape(X_train,(X_train.shape[0],1,X_train.shape[1]))
   X_test=np.reshape(X_test,(X_test.shape[0],1,X_test.shape[1]))
   
# define baseline model
def baseline_model():
   model = Sequential()
   if lstm==1:
       #  For  LSTM model
       model.add(LSTM(256, batch_input_shape=(None, 1, X_train.shape[2]),dropout=0.1, return_sequences=True))  # the following two LSTM lines go together
       #model.add(LSTM(128, dropout=0.4, recurrent_dropout=0.4))
       model.add(GlobalAveragePooling1D())
       model.add(Dropout(0.2))
       model.add(Dense(num_classes, activation='sigmoid'))
       model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

   else:
       # NOTE THAT DROPOUT AFFECTS THE ACCURACY OF THE TRAINING SET AS WELL AS THE CONNECTIONS ARE MISSING
       # DURING THE SUBSEQUENT ITERATION OF THE MODEL RESULTING IN LESS TRAINING ACCURACY IF THE DROPOUT IS TOO
       # HIGH. TO DECOUPLE TRAINING FROM TESING DROPOUT, USE THE DROPOUT TECHNIQUE ABOVE AFTER FFT
       model.add(Dense(int(num_pixels/2), input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
       #model.add(Dropout(0.2))
       #model.add(Dense(100, kernel_initializer='normal',activation='relu'))
       #model.add(Dropout(0.2))
       #model.add(Dense(100, kernel_initializer='normal',activation='relu'))
       #model.add(Dropout(0.2))
       model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
       model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
   return model
# build the model
model = baseline_model()
# Fit the model
history=model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=110, batch_size=1024, verbose=1)
#history=model.fit(X_train, y_train, validation_split=0.33, epochs=20, batch_size=32, verbose=1)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train' , 'test' ], loc= 'upper left' )
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train' ,  'test' ], loc= 'upper left' )
plt.show()
