import cv2 as cv
import numpy as np
#image capture
"""img= cv.imread("C:/Users/saswa/OneDrive/Desktop/SOME PHOTOS/IMG_0274.JPG")
cv.imshow('cat', img)
cv.waitKey(0)"""
#videocapture
'''video= cv.VideoCapture(0)                           #0 represents the webcam of the laptop
while True:
    isTrue, frame= video.read()
    cv.imshow('video', frame)
    if cv.waitKey(0)&0xFF==ord('a'):                     #cv.waitkey() returns the int value of the key pressed and ord('a') returns the int value of a
        break

video.release()
cv.destroyAllWindows()'''

#rescaling
#this will work for videos, images and live captures
def rescale_factor(frame, scale):
    width= int(frame.shape[1]*scale)
    height= int(frame.shape[0]*scale)
    dimensions= (width, height)
    return cv.resize(frame, dimensions, interpolation= cv.INTER_AREA)
img= cv.imread('C:/Users/saswa/saswata/Machine Learning A-Z (Codes and Datasets)/Part 8 - Deep Learning/Section 40 - Convolutional Neural Networks (CNN)/Python/dataset/training_set/cats/cat.45.jpg')
'''img_resize= rescale_factor(img, 0.18)
cv.imshow('org', img_resize)
cv.waitKey(0)'''

#grayscale conversion
'''img_gray_resize= rescale_factor(cv.cvtColor(img, cv.COLOR_BGR2GRAY), 0.20)
cv.imshow('gray_img',img_gray_resize)
cv.waitKey(0)'''

#blur
'''img_blur_rescale= rescale_factor(cv.GaussianBlur(img, (9,9), cv.BORDER_DEFAULT), 1)                       #5by5 kernal will decrease the pixel value by 25.
cv.imshow('Blur', img_blur_rescale)
cv.waitKey(0)'''

#edge detection
img_canny= cv.Canny(img, 125, 175)
'''cv.imshow('Canny', img_canny) 
cv.waitKey(0)'''

#Dilating an image                      #Dilating refers to adding whitespace on the edges of the image
dilate= cv.dilate(img_canny, (5,5), iterations=2)
#cv.imshow('org',img_canny)
'''cv.imshow('dilate',dilate)
cv.waitKey(0)'''

#eroding an image                       #Eroding refers to removing whitespace on the edges of the image
erode= cv.erode(dilate, (5,5), iterations= 2)
'''cv.imshow('erode',  erode)
cv.waitKey(0)'''

#transformation of image. It means to shift an image along left, right or top, down
def translate(x,y):
    transMat= np.float32([[1,0,x],[0,1,y]])
    dimesions= (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dimesions)      #here we are performing affine transformation on the image, which is basically transforming the coordinates of the pixel.
'''
-x--->left
-y--->up
x--->right
y--->down
'''
translated= translate(100,50)
'''cv.imshow('Trans', translated)
cv.waitKey(0)  '''
def rotate(img, angle, rotPoint=None):
    (height, width) = img.shape[:2]
    if rotPoint==None:
        rotPoint= (width//2, height//2)             #//2 refers to floored division by 2
    rotMat= cv.getRotationMatrix2D(rotPoint, angle, 1.0)          #1.0 is scale 
    dimensions=(width, height)
    return cv.warpAffine(img, rotMat, dimensions)
rotImg= rotate(img, 60)             #angle in degrees
'''cv.imshow('RotImg', rotImg)
cv.waitKey(0)'''

#Flipping an image
'''cv.imshow('Flip', cv.flip(img, -1))        
0 means along x axis
1 means y axis
-1 means both the axes
cv.waitKey(0)'''
img_gray= cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh= cv.threshold(img_gray, 125, 255, cv.THRESH_BINARY)
contours, hierarchies= cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
img_blur= cv.GaussianBlur(img_gray, (5,5), cv.BORDER_DEFAULT)
img_blur_canny= cv.Canny(img_blur, 125, 175) 
'''cv.imshow('BLUR_CANNY', img_blur_canny)
cv.waitKey(0)'''
'''cv.imshow('Thresh', thresh)
cv.waitKey(0)
print(len(contours))'''
blank= np.zeros(img.shape, dtype= 'uint8')
cv.drawContours(blank, contours, -1, (255,255,255), 1)                 #Draw the contours on the blank. -1 for all the contours and (255,0,0) for the contour color. 2 for the thickness
'''cv.imshow('DrawnC', blank)
cv.imshow('Thresh', thresh)'''
img_gray= cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#bgr to hsv. hsv is hue, saturation, value. hsv is how humans think and conceive color
hsv= cv.cvtColor(img, cv.COLOR_BGR2HSV)
#bgr to lab
lab= cv.cvtColor(img, cv.COLOR_BGR2LAB)
import matplotlib.pyplot as plt
'''plt.imshow(img)
plt.show()'''
#opencv always perceive image as BGR image whereas universally RGB is accepted. So the color inversion was observed.

#converting bgr to rgb
rgb= cv.cvtColor(img, cv.COLOR_BGR2RGB)
# grayscale cant be converted to hsv. for that gray->bgr->hsv

#splitting color channels into blue, green and red
b, g, r= cv.split(img)
blank2= np.zeros(img.shape[:2], dtype= 'uint8')
'''
blue= cv.merge([b, blank2, blank2])
green= cv.merge([blank2, g, blank2])
red= cv.merge([blank2, blank2, r])
'''
#for showing the different color channels

#Average blurring. 
'''
So here let the kernal window be (2k+1)*(2k+1) so the intensity of the pixel at the center of the matrix
will be the average intensity of all the surrounding pixels. The stride is 1 for the shifting of the window
'''
avg_blur= cv.blur(img, (7,7))               
#print(img.shape)
#cv.imshow('blur', avg_blur)

#Gaussian Blurring
'''
It uses weighted average of the surrounding pixels. In a Gaussian blur,
the pixels nearest the center of the kernel are given more weight than those far away from the center.
'''
gauss_blur= cv.GaussianBlur(img, (5,5), 0)

#Median Blurring
'''
Same as average blurring but here we take the median instead of the averagem More effective in reducing noises in the image 
compared to average blurring.
'''
median_blur= cv.medianBlur(img, 7)              #no need to input a tuple for the kernel size

#bitwise operator
blank3= np.zeros((400,400), dtype= 'uint8')
rec= cv.rectangle(blank3.copy(), (0,0), (100,100), (255), -1)        #-1 means to fill the rectangle
cir= cv.circle(blank3.copy(), (200,200), 150, 255, -1)
'''cv.imshow('Rec',rec)
cv.imshow('Cir', cir)'''

bit_or= cv.bitwise_or(rec, cir)
bit_and= cv.bitwise_and(rec, cir)
bit_xor= cv.bitwise_xor(rec, cir)

#masking of an image
blank4= np.zeros(img.shape[:2], dtype= 'uint8')
cir_mask= cv.circle(blank4, (img.shape[1]//2, img.shape[0]//2), 150, 255, -1)
masked= cv.bitwise_and(img, img, mask= cir_mask)
#cv.imshow('mask', masked) 

#GrayScale Histogram
gray_img= cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray_mask= cv.cvtColor(masked, cv.COLOR_BGR2GRAY)
#cv.imshow('gray', gray_mask)
gray_hist= cv.calcHist([gray_mask], [0], None, [300], [0,300])     #here [0] represents grayscale cuz grayscale has only 1 channel
'''
plt.figure('Gray_Hist')
plt.xlabel(('Bins'))                  #Bins basically represent pixel intensity
plt.ylabel('# of pixels')
plt.plot(gray_hist)
plt.xlim([0, 255])
plt.show()
'''

#Colored Histogram
colors= ('b', 'g', 'r')
'''
plt.figure('Color_Hist')
plt.xlabel(('Bins'))                  #Bins basically represent pixel intensity
plt.ylabel('# of pixels')
plt.xlim([0, 300])
for ind, colo in enumerate(colors):                #enumerate reurns list of tuples of order (index, element) of the list
    color_hist= cv.calcHist([img], [ind], cir_mask, [300], [0, 300])
    plt.plot(color_hist, color= colo)
plt.show()
'''

#Simple Thresholding
'''
Here we binarise the intensity of the pixels of the image. 
If its below the threshold then its intensity is 0/black and above the threshold then its set to 255/white.
'''
threshold, thresh= cv.threshold(gray_img, 120, 255, cv.THRESH_BINARY)          #Here, intensity above 120 will be set to 255 and below to 0
#cv.imshow('SimpleThresh', thresh)
threshold, thresh_inv= cv.threshold(gray_img, 120, 155, cv.THRESH_BINARY_INV)

#Adaptive Thresholding
adaptive_thresh_mean= cv.adaptiveThreshold(gray_img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 13, 2)
#cv.imshow('MeanAdapiveThresh', adaptive_thresh_mean)
adaptive_thresh_gaussian= cv.adaptiveThreshold(gray_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 13, 2)
#cv.imshow('GaussAdaptiveThresh', adaptive_thresh_gaussian)
'''
ADAPTIVE_THRESH_MEAN_C: threshold value is the mean of neighborhood area.
ADAPTIVE_THRESH_GAUSSIAN_C : threshold value is the weighted sum of neighborhood values where weights are a Gaussian window.
'''

#Edge Detection
#Laplacian 
lap= cv.Laplacian(gray_img, cv.CV_64F)
lap= np.uint8(np.absolute(lap))
#cv.imshow('lap', lap)

#Sobel
sobel_x= cv.Sobel(gray_img, cv.CV_64F, 1, 0)
sobel_y= cv.Sobel(gray_img, cv.CV_64F, 0, 1)
# cv.imshow('x', sobel_x)
# cv.imshow('y', sobel_y)
sobel_combined= cv.bitwise_or(sobel_x, sobel_y)

cv.waitKey(0)