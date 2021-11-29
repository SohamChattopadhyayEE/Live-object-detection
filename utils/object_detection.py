#https://github.com/darshanadakane/yolov3_realTimeObjectDetection/blob/master/YOLO_ObjectDetectionInVideo_git.ipynb
#from _typeshed import Self
import numpy as np
import cv2
import time
import argparse

class object_detection() :
    def __init__(self, model_weights, model_cfg, dataset_classes,is_realtime,
                    vid_path, frame_dimension, threshold_confidence):
        self.model_weights = model_weights
        self.model_cfg = model_cfg
        self.dataset_classes = dataset_classes
        self.is_realtime = is_realtime
        self.vid_path = vid_path
        self.frame_dimension = frame_dimension
        self.threshold_confidence= threshold_confidence

    def detect_object(self):
        
        net = cv2.dnn.readNet(self.model_weights,self.model_cfg)

        classes = []

        with open(self.dataset_classes, 'r') as f :
            classes = f.read().splitlines()
        print("Classes : ", classes)

        # Layers
        layer_names = net.getLayerNames()
        outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        colors= np.random.uniform(0,255,size=(len(classes),3))

        # Video capturing
        print('-----------------------------------------')
        if self.is_realtime : 
            print("Real time object detection is being performed")
            cap = cv2.VideoCapture(0)
        else :
            print("Video object detection is being performed")
            cap = cv2.VideoCapture(self.video_path)
        font = cv2.FONT_HERSHEY_PLAIN
        frame_id = 0
        starting_time= time.time()

        while True:
            _,frame= cap.read() # 
            frame_id+=1
            
            height,width,channels = frame.shape
            #detecting objects
            blob = cv2.dnn.blobFromImage(frame,0.00392,(self.frame_dimension,self.frame_dimension),
                (0,0,0),True,crop=False) #reduce 416 to 320  

                
            net.setInput(blob)
            outs = net.forward(outputlayers)
            #print(outs[1])


            #Showing info on screen/ get confidence score of algorithm in detecting an object in blob
            class_ids=[]
            confidences=[]
            boxes=[]
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > self.threshold_confidence:
                        #object detected
                        center_x= int(detection[0]*width)
                        center_y= int(detection[1]*height)
                        w = int(detection[2]*width)
                        h = int(detection[3]*height)

                        #cv2.circle(img,(center_x,center_y),10,(0,255,0),2)
                        #rectangle co-ordinaters
                        x=int(center_x - w/2)
                        y=int(center_y - h/2)
                        #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

                        boxes.append([x,y,w,h]) #put all rectangle areas
                        confidences.append(float(confidence)) #how confidence was that object detected and show that percentage
                        class_ids.append(class_id) #name of the object tha was detected

            indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.4,0.6)
            #print("indexes: ",indexes)


            for i in range(len(boxes)):
                if i in indexes:
                    x,y,w,h = boxes[i]
                    label = str(classes[class_ids[i]])
                    confidence= confidences[i]
                    color = colors[class_ids[i]]
                    cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
                    cv2.putText(frame,label+" "+str(round(confidence,2)),(x,y+30),font,1,(20,50,255),2)
                    

            elapsed_time = time.time() - starting_time
            fps=frame_id/elapsed_time
            cv2.putText(frame,"FPS:"+str(round(fps,2)),(10,50),font,2,(0,0,0),1)
            
            #cv2.imshow("Image",frame)
            #key = cv2.waitKey(1)
            ret,buffer=cv2.imencode('.jpg', frame)
            frame=buffer.tobytes()
            yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n')
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        #cap.release()    
        #cv2.destroyAllWindows()

if __name__ == "__main__":
    obj = object_detection()
    obj.detect_object()