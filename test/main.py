import cv2
import time
import os
#cam = cv2.VideoCapture(0)
#cv2.namedWindow("frame");
#cv2.resizeWindow("frame", 300,200);
#count:int = 0
#hand = "./datasets/aGest.xml"
#hand_tracker = cv2.CascadeClassifier(hand)
#palm = "./datasets/palm.xml"
#palm_tracker = cv2.CascadeClassifier(palm)
# Source data

def getAbsoluteFilePathsInDir(dir:str) -> list:
    output:list = []
    for root, _, files in os.walk(dir):
        for file in files:
            output.append(os.path.join(root, file))
	
    return output
    
def getTrackers(datasetsPaths:list) -> list:
    return [cv2.CascadeClassifier(datasetPath) for datasetPath in datasetsPaths]

def testDatasets(imagesPaths, datasetsPaths, trackers):
    
    for count, tracker in enumerate(trackers):
        trackerName = datasetsPaths[count][:-4]
        print(trackerName)
        os.mkdir(trackerName) if not os.path.exists(trackerName) else None
        
        for imageNumber, imagePath in enumerate(imagesPaths):
            image = cv2.imread(imagePath)
            grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            objects = tracker.detectMultiScale(grayImage)
            if len(objects):
                for (x,y,w,h) in objects:
                    cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)

                cv2.imwrite(os.path.join(trackerName, f"{imageNumber}_image.png"), image)
                
def main():
    datasetsPaths:list = getAbsoluteFilePathsInDir(os.path.abspath("datasets"))
    imagesPaths:list = getAbsoluteFilePathsInDir(os.path.abspath("test_images"))
    trackers:list = getTrackers(datasetsPaths)
    testDatasets(imagesPaths, datasetsPaths, trackers)

        #_, image = cam.read()
        ##image = cv2.resize(image, (200, 100)) 
        ## convert color image to grey image
        #gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ##gray_img = image
#
        #hands = hand_tracker.detectMultiScale(gray_img)
        #palms = palm_tracker.detectMultiScale(gray_img)
#
        ## display the coordinates of different hands - multi dimensional array
#
        ## draw rectangle around the hands
        #for (x,y,w,h) in hands:
        #    cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)
        #    #cv2.putText(image, 'Car', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#
        ## draw rectangle around the palms
        #for (x,y,w,h) in palms:
        #    cv2.rectangle(image, (x,y), (x+w, y+h), (0,0,255), 2)
        #    #cv2.putText(image, 'Human', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # Fi#nally display the image with the markings
        #count+=1
        #cv2.imwrite(f"./images/file_{count}.png", image)
    #plt.imshow(image)
    #plt.show()
    # wait for the keystroke to exit

if __name__ == "__main__":
    main()

