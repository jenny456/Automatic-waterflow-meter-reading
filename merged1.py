############import necessary files################
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import operator
import os
from PIL import Image
import pytesseract
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Flatten, Dense
import sys
import pandas as pd
import datetime
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
################################################
image = cv2.imread('use5.jpeg')
image111=cv2.resize(image,(300,400))
image = imutils.resize(image, width=300)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# apply a tophat (whitehat) morphological operator to find light
# regions against a dark background (i.e., the credit card numbers)
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)

# compute the Scharr gradient of the tophat image, then scale
# the rest back into the range [0, 255]
gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0,
	ksize=-1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
gradX = gradX.astype("uint8")

# apply a closing operation using the rectangular kernel to help
# cloes gaps in between digits, then apply
# Otsu's thresholding method to binarize the image
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
cv2.imwrite('closing.jpeg',gradX)
thresh = cv2.threshold(gradX, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# apply a second closing operation to the binary image, again
# to help close gaps between number 
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
kernel = np.ones((5,5),np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

thresh = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, rectKernel)
cv2.imwrite('finalclosing.jpeg',thresh)

#Finding contours
cnts=cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
locs = []

# loop over the contours
for (i, c) in enumerate(cnts):
    # compute the bounding box of the contour, then use the
    # bounding box coordinates to derive the aspect ratio
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)
    if ar >4 and ar < 8:
        # append the bounding box region of the digits group
	# to our locations list
        if (w >50 and w < 100) and (h > 15 and h < 30):
            # append the bounding box region of the digits group
	    # to our locations list
            locs.append((x, y, w, h))
            #print(w,h,ar)

#drawing rectangle around the detected region            
for (i,(x,y,w,h)) in enumerate(locs):
        cv2.rectangle(image, (x - 5, y - 5),(x+w + 5,y + h + 5), (0, 0, 255), 2)

#cv2.imshow('check',image)
cv2.imwrite('detected.jpeg',image)

im11=cv2.imread('detected.jpeg')

#cropping the detecting part and saving
crop_image=im11[y:y+h, x:x+w+5]
cv2.imwrite('crop_usenew.jpeg',crop_image)
#cv2.imshow('crp',crop_image)

#############################################filtering#############################
#print('cv2 version: %s' % cv2._version_)
img_orig = cv2.imread('crop_usenew.jpeg')

img=cv2.resize(img_orig,(300,100))
cv2.imwrite('resize.jpeg',img)
img_crp = cv2.imread('resize.jpeg')
img1 = cv2.imread('resize.jpeg')
# img_orig = cv2.imread('/Users/yamato/OneDrive/WaterMeterLogs/test-new.jpg')
# img_orig = cv2.imread('/Users/yamato/OneDrive/WaterMeterLogs/ocrerror02.jpg')

#cv2.imshow('resize',img1)
img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img.shape, img.dtype


th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 115, 1)
#cv2.imshow("ada",th)
kernel = np.ones((3,3),np.uint8)

closing = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
#cv2.imshow("clo_ada",closing)

ret,thresh2 = cv2.threshold(closing,110,255,cv2.THRESH_BINARY_INV)
#cv2.imshow('ooo.jpeg',thresh2)


# Canny edge detection# Canny 
edges = cv2.Canny(closing, 200, 250, apertureSize=3, L2gradient=True)
# edges = cv2.5Canny(img, 235, 250)
#plt.imshow(edges, cmap='gray')
#cv2.imshow('canny',edges)
image, contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours_dict = dict()

for cont in contours:
    x, y, w, h = cv2.boundingRect(cont)
    area = cv2.contourArea(cont)
    if 10 < area  and 10 < w and w<105 and h <101:
        contours_dict[(x, y, w, h)] = cont
        #print(area,w,h)
contours_filtered = sorted(contours_dict.values(), key=cv2.boundingRect)
#cv2.imshow('img_contours1',contours_filtered)
#print("hey")
blank_background = np.zeros_like(edges)
img_contours = cv2.drawContours(blank_background, contours_filtered,-1, (255,255,255), thickness=2)
#img_contours = cv2.drawContours(blank_background, contours_filtered, -1, (255,255,255), thickness=cv2.FILLED)
#plt.axis('off')
#plt.imshow(img_contours, 'gray')
cv2.imwrite('cnttry4.jpeg',img_contours)
cv2.imshow('img_contours',img_contours)
imc=cv2.imread('cnttry4.jpeg')
###########################detection#########################################
img_width, img_height = 28, 28

def create_model():
  model = Sequential()

  model.add(Convolution2D(16, 5, 5, activation='relu', input_shape=(img_width, img_height, 3)))
  model.add(MaxPooling2D(2, 2))

  model.add(Convolution2D(32, 5, 5, activation='relu'))
  model.add(MaxPooling2D(2, 2))

  model.add(Flatten())
  model.add(Dense(1000, activation='relu'))

  model.add(Dense(10, activation='softmax'))

  #model.summary()

  return model
res=[]
crop_image1=img_contours[y:120, x:x+55]
cv2.imwrite('dig1.png',crop_image1)
#num_detectionFunction(crop_image1)
img = cv2.imread("dig1.png")
img = cv2.resize(img, (img_width, img_height))
model = create_model()
model.load_weights('./mnistneuralnet.h5')
arr = np.array(img).reshape((img_width,img_height,3))
arr = np.expand_dims(arr, axis=0)
prediction = model.predict(arr)[0]
bestclass = ''
bestconf = -1
for n in [0,1,2,3,4,5,6,7,8,9]:
	if (prediction[n] > bestconf):
		bestclass = str(n)
		bestconf = prediction[n]
#print('I think this digit is a ' + bestclass + ' with ' + str(bestconf * 100) + '% confidence.')
#print(bestclass)
#res[0]=bestclass
res.append(bestclass)
crop_image2=img_contours[y:120, x+55:x+115]
cv2.imwrite('dig2.png',crop_image2)
#num_detectionFunction(crop_image2)
img = cv2.imread("dig2.png")
img = cv2.resize(img, (img_width, img_height))
model = create_model()
model.load_weights('./mnistneuralnet.h5')
arr = np.array(img).reshape((img_width,img_height,3))
arr = np.expand_dims(arr, axis=0)
prediction = model.predict(arr)[0]
bestclass = ''
bestconf = -1
for n in [0,1,2,3,4,5,6,7,8,9]:
	if (prediction[n] > bestconf):
		bestclass = str(n)
		bestconf = prediction[n]
#print('I think this digit is a ' + bestclass + ' with ' + str(bestconf * 100) + '% confidence.')
#print(bestclass)
#res[1]=bestclass
res.append(bestclass)
crop_image3=img_contours[y:120, x+115:x+170]
cv2.imwrite('dig3.png',crop_image3)
#num_detectionFunction(crop_image3)
img = cv2.imread("dig3.png")
img = cv2.resize(img, (img_width, img_height))
model = create_model()
model.load_weights('./mnistneuralnet.h5')
arr = np.array(img).reshape((img_width,img_height,3))
arr = np.expand_dims(arr, axis=0)
prediction = model.predict(arr)[0]
bestclass = ''
bestconf = -1
for n in [0,1,2,3,4,5,6,7,8,9]:
	if (prediction[n] > bestconf):
		bestclass = str(n)
		bestconf = prediction[n]
#print('I think this digit is a ' + bestclass + ' with ' + str(bestconf * 100) + '% confidence.')
#print(bestclass)
#res[2]=bestclass
res.append(bestclass)
crop_image4=img_contours[y:120, x+175:x+230]
cv2.imwrite('dig4.png',crop_image4)
#num_detectionFunction(crop_image4)
img = cv2.imread("dig4.png")
img = cv2.resize(img, (img_width, img_height))
model = create_model()
model.load_weights('./mnistneuralnet.h5')
arr = np.array(img).reshape((img_width,img_height,3))
arr = np.expand_dims(arr, axis=0)
prediction = model.predict(arr)[0]
bestclass = ''
bestconf = -1
for n in [0,1,2,3,4,5,6,7,8,9]:
	if (prediction[n] > bestconf):
		bestclass = str(n)
		bestconf = prediction[n]
#print('I think this digit is a ' + bestclass + ' with ' + str(bestconf * 100) + '% confidence.')
#print(bestclass)
#res[3]=bestclass
res.append(bestclass)
crop_image5=img_contours[y:120, x+230:x+290]
cv2.imwrite('dig5.png',crop_image5)
#num_detectionFunction(crop_image5)
img = cv2.imread("dig5.png")
img = cv2.resize(img, (img_width, img_height))
model = create_model()
model.load_weights('./mnistneuralnet.h5')
arr = np.array(img).reshape((img_width,img_height,3))
arr = np.expand_dims(arr, axis=0)
prediction = model.predict(arr)[0]
bestclass = ''
bestconf = -1
for n in [0,1,2,3,4,5,6,7,8,9]:
	if (prediction[n] > bestconf):
		bestclass = str(n)
		bestconf = prediction[n]
#print('I think this digit is a ' + bestclass + ' with ' + str(bestconf * 100) + '% confidence.')
#print(bestclass)
#res[4]=bestclass
res.append(bestclass)
myarray = np.asarray(res)
#print(myarray)
#print(myarray[3])
array1=''.join(myarray)
print(array1)
#####################################csv updation##############################
df1=pd.read_csv('results.csv')
now=datetime.datetime.now()
date=now.strftime("%d-%m-%Y")
time=now.strftime("%H:%M")
output=array1
df2=pd.DataFrame([[time,date,output]],columns=['Time','Date','Output'])
print(df2.head())
df1=pd.concat([df1,df2])
df1.to_csv('results.csv',index=False)
##########################mailing_of_csv#############
'''fromaddr = "from@gmail.com"
toaddr = "to@gmail.com"
msg = MIMEMultipart()
msg['From'] = fromaddr
msg['To'] = toaddr
msg['Subject'] = "subject"
body = "Body"
msg.attach(MIMEText(body, 'plain'))
filename = "results.csv"
attachment = open("path of csv file", "rb")
p = MIMEBase('application', 'octet-stream')
p.set_payload((attachment).read())
encoders.encode_base64(p)
p.add_header('Content-Disposition', "attachment; filename= %s" % filename)
msg.attach(p)
s = smtplib.SMTP('smtp.gmail.com', 587)
s.starttls()
s.login(fromaddr, "password")
text = msg.as_string()
s.sendmail(fromaddr, toaddr, text)
s.quit()'''

