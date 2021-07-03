import cv2
import numpy as np
import matplotlib.pyplot as plt 

model = cv2.dnn.readNet('yolov3_face_mask.weights','yolov3.cfg')

claases = []

with open('obj.names', 'r') as f:
    claases = [line.strip() for line in f.readlines()]

layers = model.getLayerNames()
outputLayers = [layers[i[0] - 1] for i in model.getUnconnectedOutLayers()]

test_img = cv2.imread('mask.jpg')
test_img = cv2.resize(test_img, None, fx=0.8, fy=0.8)
height, width, _ = test_img.shape

features = cv2.dnn.blobFromImage(test_img, 0.00392, (608, 608), (0,0,0), True, crop=False)

model.setInput(features)
output = model.forward(outputLayers)

boxes = []
confidences = []
class_ids = []

claases = []

with open('obj.names', 'r') as f:
    claases = [line.strip() for line in f.readlines()]

#  Get the name of all layers of the network
# ============================================
layers = model.getLayerNames()
outputLayers = [layers[i[0] - 1] for i in model.getUnconnectedOutLayers()]

for o in output:
    for detection in o:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            centerX = int(detection[0] * width)
            centerY = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            
            # rectangle coordinates from center
            # =================================
            x, y = int(centerX - w / 2), int(centerY - h / 2) 
            
            boxes.append([x,y,w,h]) # pass coordinates
            class_ids.append(class_id) # pass corresponding class
            confidences.append(float(confidence)) # pass corresponding confidence class
    
# draw rectangle & put label in each feature in image
# ===================================================
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
for i in range(len(boxes)):
    if i in indices:
        x,y,w,h = boxes[i]
        label = claases[class_ids[i]]
        confidence = round(confidences[i] * 100, 2)
        cv2.rectangle(test_img, (x,y), (x + w, y + h), (0,0,255), 3)
        cv2.putText(test_img, label + ' ' + str(confidence) + '%', (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 1, cv2.LINE_AA)
        
cv2.imshow('Mask detection', test_img)
cv2.imwrite('mask.png', test_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

