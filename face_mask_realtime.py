import cv2


"""
Sometimes It seems a problem with brightness. Try to experiment with turn on/off the electric light in sunlight in the room.
"""
print("Sometimes It seems a problem with brightness. Try to experiment with turn on/off the electric light in sunlight in the room.")

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
mask_cascade = cv2.CascadeClassifier('mask_cascade.xml')

def detect_mask(face):
    mask = mask_cascade.detectMultiScale(face, 1.01, 11)
    return mask


def detect_face(originalImage):
   is_mask = ''
   faces = face_cascade.detectMultiScale(originalImage, 1.3, 5)  # for usb camera
   for(x, y, w, h) in faces: 
         only_face = originalImage[y:y+h, x:x+w]
         mask = detect_mask(only_face)
         if len(mask) > 0:
             is_mask='Mask'
         else:
             is_mask='No Mask'
         wholeFace = cv2.rectangle(originalImage, (x, y), (x + w, y + h), (255, 0, 0), 2  )
         cv2.putText(wholeFace, is_mask, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
           
   return originalImage

video_capture = cv2.VideoCapture(0)

while True:
    
    ret, originalImage = video_capture.read()
    if not ret:
        break
    
    canvas = detect_face(originalImage)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
       
video_capture.release()
cv2.destroyAllWindows()
