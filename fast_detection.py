import cv2
import os

class WebCam(cv2.VideoCapture):
    def __init__(self, frameName:str = "test_frame"):
        super().__init__(1)
        self.set(3,1280)
        self.set(4,720)
        self.set(10,70)    
        cv2.namedWindow(frameName)
        self.frameName = frameName

class NetModel(cv2.dnn_DetectionModel):
    def __init__(self, weightsPath:str, configPath:str, classNamesPath:str, thres:float) -> None:
        super().__init__(weightsPath, configPath)
        self.setInputSize(320,320)
        self.setInputScale(1.0/ 127.5)
        self.setInputMean((127.5, 127.5, 127.5))
        self.setInputSwapRB(True)
        self.thres:float = 0.45
        self.weightsPath:str = weightsPath    
        self.configPath:str = configPath    
        self.classNamesPath:str = classNamesPath
        self.classNames:list = self.readClassNames()    
        self.classNamesLen:int = len(self.classNames)
        print(self.classNamesLen    )

    def readClassNames(self) -> None:
        return open(self.classNamesPath, "rt", encoding="UTF-8").read().rstrip("\n").split("\n")

    def detectObjectsInImage(self, image) -> object:
            classIds, confs, bbox = self.detect(image,confThreshold=self.thres)
            if len(classIds):
                for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
                    cv2.rectangle(image,box,color=(0,255,0),thickness=2)
                    if classId and classId <= self.classNamesLen:
                        cv2.putText(image,self.classNames[classId-1].upper(),(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    def detectObjectsInWebCam(self, frameName:str, Webcam:WebCam) -> None:
        while True:
            _, img = Webcam.read()
            self.detectObjectsInImage(img)
            cv2.imshow(frameName, img)
            cv2.waitKey(1)
            

if __name__ == "__main__":
    configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
    weightsPath = "frozen_inference_graph.pb"
    classIdsPath = "coco.names"
    net = NetModel(weightsPath, configPath, classIdsPath, thres=0.45)
    cam = WebCam()
    net.readClassNames()
    net.detectObjectsInWebCam(cam.frameName, cam)
    
#cap = cv2.VideoCapture(1)

#classNames= []
#with open("coco.names","r", encoding="UTF-8") as file:
#    classNames = file.read().rstrip("n").split("n")
# 
#

#net = cv2.dnn_DetectionModel(weightsPath,configPath)
#net.setInputSize(320,320)
#net.setInputScale(1.0/ 127.5)
#net.setInputMean((127.5, 127.5, 127.5))
#net.setInputSwapRB(True)

#count = 0
#classNames = net.getClassNames()
##for root, _, files in os.walk("test_images"):
#while True:
#    success,img = cam.read()
#    #for file in files:
#       #with open(os.path.join(root, file), "rt", encoding="UTF-8") as file:
#            #img = cv2.imread(os.path.join(root, file))
#    classIds, confs, bbox = net.detect(img,confThreshold=thres)
#    if len(classIds) != 0:
#        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
#            cv2.rectangle(img,box,color=(0,255,0),thickness=2)
#            if classId <= 30:
#                cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
#                cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
#                    #print(f"Done image number: {count}")
#                    #cv2.imwrite(os.path.join(os.getcwd(), f"output_images\{count}_image.png"), img)
#                    #count+=1
#                #cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
#                #cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
# 
#    cv2.imshow("Output",img)
#    cv2.waitKey(1)