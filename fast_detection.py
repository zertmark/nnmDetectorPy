import cv2
import os
import numpy as np

class WebCam(cv2.VideoCapture):
    def __init__(self, frameName:str = "test_frame"):
        super().__init__(1)
        self.set(3,1280)
        self.set(4,720)
        self.set(10,70)    
        cv2.namedWindow(frameName)
        self.frameName = frameName

class NetModel(cv2.dnn_DetectionModel):
    def __init__(self, weightsPath:str, configPath:str, classNamesPath:str, thres:float = 0.45, nmsUse:bool = False) -> None:
        super().__init__(weightsPath, configPath)
        self.setInputSize(320,320)
        self.setInputScale(1.0/ 127.5)
        self.setInputMean((127.5, 127.5, 127.5))
        self.setInputSwapRB(True)
        self.thres:float = 0.45
        self.nms_threshold = 0.2
        self.weightsPath:str = weightsPath    
        self.configPath:str = configPath    
        self.classNamesPath:str = classNamesPath
        self.classNames:list = self.readClassNames()    
        self.classNamesLen:int = len(self.classNames)
        self.nmsUse = nmsUse

    def readClassNames(self) -> list[str]:
        return open(self.classNamesPath, "rt", encoding="UTF-8").read().rstrip("\n").split("\n")

    def getBoxesIDsNMS(self, bbox, confs) -> list[int]:
        return cv2.dnn.NMSBoxes(list(bbox), list(map(float, list(np.array(confs).reshape(1,-1)[0]))), self.thres, self.nms_threshold)

    def detectObjectsInImage(self, ImageCV) -> object:
            classIds, confs, bbox = self.detect(ImageCV,confThreshold=self.thres)
            bboxIDs:list = self.getBoxesIDsNMS(bbox, confs) if self.nmsUse else range(len(bbox))
            if len(classIds):
                for classId, confidence,boxId in zip(classIds.flatten(),confs.flatten(),bboxIDs):
                    cv2.rectangle(ImageCV,bbox[boxId],color=(0,255,0),thickness=2)
                    cv2.putText(ImageCV,self.classNames[classId-1].upper(),(bbox[boxId][0]+10,bbox[boxId][1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            
            return ImageCV

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
    net = NetModel(weightsPath, configPath, classIdsPath, thres=0.45, nmsUse=True)
    cam = WebCam("Testing NMS")
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