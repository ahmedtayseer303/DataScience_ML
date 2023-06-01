# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 17:40:36 2022

@author: Tayseer
"""
import cv2
import numpy as np
import time
# import matplotlib.pyplot as plt

weights  = r"C:\Users\Tayseer\Documents\Senior2\Graduation project\models\YOLOv3-tiny\yolov3-tiny.weights"
cfg  = r"C:\Users\Tayseer\Documents\Senior2\Graduation project\models\YOLOv3-tiny\yolov3-tiny.cfg"

# have most accuracy
weights2 = r"C:\Users\Tayseer\Documents\Senior2\Graduation project\models\YOLOv3-608\yolov3.weights"
cfg2 = r"C:\Users\Tayseer\Documents\Senior2\Graduation project\models\YOLOv3-608\yolov3.cfg"

# Tiny yolo
weights3 = r"C:\Users\Tayseer\Documents\Senior2\Graduation project\models\Tiny YOLO\yolov2-tiny.weights"
cfg3 = r"C:\Users\Tayseer\Documents\Senior2\Graduation project\models\Tiny YOLO\yolov2-tiny.cfg"

img_path  = r"C:\Users\Tayseer\Documents\Senior2\Graduation project\models\YOLOv3-tiny\image1.jpg"
coco = r"C:\Users\Tayseer\Documents\Senior2\Graduation project\models\YOLOv3-tiny\coco.names"

net = cv2.dnn.readNet(weights, cfg)
#%%

# Read 80 class names of coco
classes = []
with open(coco,'r') as f:
    classes = f.read().splitlines()
# print(len(classes))

#%%
def detect_img(img,is_video=False):
    start = time.time()
    if not is_video:
        img = cv2.imread(img)
    height, width, _ =img.shape
    
    # will blob of 3 images for 3 channels
    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Getting output layers names (to get output from more than one layer) and pass names into frword functions
    output_layers_names = net.getUnconnectedOutLayersNames()

    print(output_layers_names)
    layerOutputs = net.forward(output_layers_names)
    print("Time After layer outputs: "+ str(time.time()-start))

    boxes = []
    confidences = []
    class_ids = []
    
    """ 
    Each detection has 85 elements. first 4 (0 - 3) for the location of the bounding box. from (5-end) 80 element
    the score for each class
    
    """
    # one for loop for layers and other one for detections in each layer
    for output in layerOutputs:
        for detection in output:
    
            # get the all 80 classese probabilites and get the max one
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                #--> multiplying for resizing
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)
    
                x = int(center_x - w/2)
                y = int(center_y - h/2)
    
                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)
    
    # print(len(boxes))
    # print(boxes)
    
    # --> There is a case of haveing 2 or more different boxes on each other
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4) # 0.2 is the same as the threashold above, last parameter is long max supresssion by default 0.4 
    
    # Showing the remaining boxes. flatten() is used to get a copy of an given array collapsed into one dimension.
    # print(indexes.flatten())
    
    font = cv2.FONT_HERSHEY_PLAIN
    # Make different random 100 colors take values from 0-255 in 3 channels
    colors = np.random.uniform(0, 255, size=(100,3))
    
    if len(indexes)>0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            
            # One color for each object
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2) # Thickness of the rectangle is 2
    
            # 2 is the text size and other one is the thickness
            cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)
    end = time.time()
    print("Time per frame: "+ str(end-start))
    return img

def input_img(img_path):
    img = detect_img(img_path)
    cv2.namedWindow("window name", cv2.WINDOW_NORMAL)
    cv2.imshow("window name", img)
    while True:
        key = cv2.waitKey(1)
        if key==27:
            break
    cv2.destroyAllWindows()


# Video input
def input_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while True:
        _, img = cap.read()
        img = detect_img(img,True)
        cv2.imshow('Image', img)
        key = cv2.waitKey(1)
        if key==27:
            break
    cap.release()
    cv2.destroyAllWindows()


def writeVideo(inVideoPath, outVideoPath):
    inVideo = cv2.VideoCapture(inVideoPath)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = 0
    length = int(inVideo.get(cv2.CAP_PROP_FRAME_COUNT))    
    print("Number of frames: ", length)
    curFrame = 0
    while(True):
        retVal, frame = inVideo.read()
        if not retVal:
            break
        if not out:
            out = cv2.VideoWriter(outVideoPath + '/outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15.0, (frame.shape[1], frame.shape[0]))
        img = detect_img(frame,True)
        out.write(img)
        curFrame += 1
        if (curFrame) % 10 == 0:
            print("Current frame: ", curFrame)
    print("Video saved successfully")
    inVideo.release()
    out.release()
    cv2.destroyAllWindows()
    
def main():
    video = r"C:\Users\Tayseer\Documents\Senior2\Graduation project\models\YOLOv3-tiny\test2.mp4"
    output = r"C:\Users\Tayseer\Documents\Senior2\Graduation project\models\YOLOv3-tiny\ouput"
    input_img(img_path)
    # input_video(video)
    # writeVideo(video,output)
main()    