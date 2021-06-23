import cv2
import numpy as np
import matplotlib.pyplot as plt 

model = cv2.dnn.readNet('yolov3_face_mask.weights','yolov3.cfg')

claases = []

with open('obj.names', 'r') as f:
    claases = [line.strip() for line in f.readlines()]

layers = model.getLayerNames()
outputLayers = [layers[i[0] - 1] for i in model.getUnconnectedOutLayers()]

"""
test_img = cv2.imread('7.jpg')
test_img = cv2.resize(test_img, None, fx=0.8, fy=0.8)
height, width, _ = test_img.shape

features = cv2.dnn.blobFromImage(test_img, 0.00392, (128, 128), (0,0,0), True, crop=False)

model.setInput(features)
output = model.forward(outputLayers)
"""

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

capture = cv2.VideoCapture(0)
#capture = cv2.VideoCapture('Woman wearing face mask and coat walking on street.mp4')
video_writer = cv2.VideoWriter( 'output.avi', 
                               cv2.VideoWriter_fourcc( *'XVID' ), capture.get(5),
                               (int( capture.get( 3 ) ), int( capture.get( 4 ) )) )

frame_no = 0
while True:
    ret, frame = capture.read()
    if not ret:
        break
    frame_no += 1
    height, width, channels = frame.shape
    features = cv2.dnn.blobFromImage(frame, scalefactor = 0.00392, size = (256,256), mean=(0,0,0),  swapRB = True, crop=False)

    # input into neural network
    # =========================
    model.setInput(features)
    output = model.forward(outputLayers)

    # showing info on the screen
    # ========================== 
    boxes = []
    confidences = []
    class_ids = []

    for o in output:
        for detection in o:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                centerX = int(detection[0] * width)
                centerY = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                #cv2.circle(test_img,(centerX, centerY), 10,(0,255,0),2)
                
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
            color = (0,0,255)
            if label == 'mask':
                color = (0, 255, 0)
            confidence = round(confidences[i] * 100, 2)
            cv2.rectangle(frame, (x,y), (x + w, y + h), color, 3)
            cv2.putText(frame, label + ' ' + str(confidence) + '%', (x, y - 8), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, color, 1, cv2.LINE_AA)
    
    cv2.imshow('object detection', frame)
    #video_writer.write(frame)
    print("frame: {}/{}".format(frame_no, capture.get(7)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()

# ************************ image ********************* #
"""
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
cv2.waitKey(0)
cv2.destroyAllWindows()
"""