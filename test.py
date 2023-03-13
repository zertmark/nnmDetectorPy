import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
#image_test = cv2.imread(cv2.samples.findFile("image.jpg"))
#if image_test is None:
#    sys.exit("Could not read the image.")
#
#plt.figure(), plt.title("Original"), plt.imshow(image_test), plt.axis("off");
#coin_blur = cv2.medianBlur(src=image_test, ksize=13)
#
#plt.figure(), plt.title("Low Pass Filtering (Blurring)"), plt.imshow(coin_blur), plt.axis("off");
#coin_gray = cv2.cvtColor(coin_blur, cv2.COLOR_BGR2GRAY)
#
#plt.figure(), plt.title("Gray Scale"), plt.imshow(coin_gray, cmap="gray"), plt.axis("off");
#ret, coin_thres = cv2.threshold(src=coin_gray, thresh=75, maxval=255, type=cv2.THRESH_BINARY)
#
#plt.figure(), plt.title("Binary Threshold"), plt.imshow(coin_thres, cmap="gray"), plt.axis("off");
#contour, hierarchy = cv2.findContours(image=coin_thres.copy(), mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_SIMPLE)
#
#for i in range(len(contour)):
#    
#    if hierarchy[0][i][3] == -1: # external contour
#        cv2.drawContours(image=image_test,contours=contour,contourIdx=i, color=(0,255,0), thickness=10)
#        
#plt.figure(figsize=(7,7)), plt.title("After Contour"), plt.imshow(image_test, cmap="gray"), 
#plt.axis("off");
# read data
def drawObjects(image, coordinates:list, text:str):
    for (x,y,w,h) in coordinates:
        cv2.rectangle
cam = cv2.VideoCapture(0)
count:int = 0
hand_class = "hand.xml"
palm_class = "palm.xml"
while True:
    r, image = cam.read()
    gray_img = cv2.cvtColor(image, cv2.COLOR_BRG2GRAY)
    hand_tracker = cv2.CascadeClassifier(hand_class)
    palm_tracker = cv2.CascadeClassifier(palm_class)
    hands = hand_tracker.detectMultiScale(gray_img)
    palms = palm_tracker.detectMultiScale(gray_img)




#    image_test = image
#    #plt.figure(), plt.title("Original"), plt.imshow(image_test), plt.axis("off");
#
#    # Blurring
#    coin_blur = cv2.medianBlur(src=image_test, ksize=15)
#    #plt.figure(), plt.title("Low Pass Filtering (Blurring)"), plt.imshow(coin_blur), plt.axis("off");
#
#    # Gray Scale
#    coin_gray = cv2.cvtColor(coin_blur, cv2.COLOR_BGR2GRAY)
#    #plt.figure(), plt.title("Gray Scale"), plt.imshow(coin_gray, cmap="gray"), plt.axis("off");
#
#    # Binary Threshold
#    ret, coin_thres = cv2.threshold(src=coin_gray, thresh=65, maxval=255, type=cv2.THRESH_BINARY)
#    #plt.figure(), plt.title("Binary Threshold"), plt.imshow(coin_thres, cmap="gray"), 
#    #plt.axis("off");
#    kernel = np.ones((3,3), np.uint8)
#
#    opening = cv2.morphologyEx(coin_thres, cv2.MORPH_OPEN, kernel=kernel, iterations=2)
#
#    #plt.figure(), plt.title("Opening"), plt.imshow(opening, cmap="gray"), plt.axis("off");
#    dist_transform = cv2.distanceTransform(src=opening, distanceType=cv2.DIST_L2, maskSize=5)
#
#    #plt.figure(), plt.title("Distance Transform"), plt.imshow(dist_transform, cmap="gray"), plt.axis("off");
#    ret, sure_foreground = cv2.threshold(src=dist_transform, thresh=0.4*np.max(dist_transform), maxval=255, type=0)
#
#    #plt.figure(), plt.title("Fore Ground"), plt.imshow(sure_foreground, cmap="gray"), plt.axis("off");
#    sure_background = cv2.dilate(src=opening, kernel=kernel, iterations=1) #int
#
#    sure_foreground = np.uint8(sure_foreground) # change its format to int
#    unknown = cv2.subtract(sure_background, sure_foreground)
#
#    #plt.figure(), plt.title("BackGround - ForeGround = "), plt.imshow(unknown, cmap="gray"), plt.axis("off");
#    ret, marker = cv2.connectedComponents(sure_foreground)
#
#    marker = marker + 1
#
#    marker[unknown == 255] = 0 # White area is turned into Black to find island for watershed
#
#    #plt.figure(), plt.title("Connection"), plt.imshow(marker, cmap="gray"), plt.axis("off");
#    marker = cv2.watershed(image=image_test, markers=marker)
#
#    #plt.figure(), plt.title("Watershed"), plt.imshow(marker, cmap="gray"), plt.axis("off");
#    contour, hierarchy = cv2.findContours(image=marker.copy(), mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_SIMPLE)
#
#    for i in range(len(contour)):
#
#        if hierarchy[0][i][3] == -1:
#            cv2.drawContours(image=image_test,contours=contour,contourIdx=i, color=(255,0,0), thickness=3)
#
#    plt.figure(figsize=(7,7))
#    plt.axis("off")
#    plt.imshow(image_test)
#    plt.imsave(f"./images/file_{count}.png", image_test)
#    count+=1
#    time.sleep(0.2)
#    plt.close()
    #cv2.imshow(cv2.cvtColor(np.asarray(figure.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR))
    