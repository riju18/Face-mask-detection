import cv2

mask_cascade = cv2.CascadeClassifier('mask_cascade.xml')

def detect(originalImage):
   mask = mask_cascade.detectMultiScale(originalImage, 1.01, 11) # img
   print(mask)
   for(x, y, w, h) in mask: 
       the_mask = cv2.rectangle(originalImage, (x, y), (x + w, y + h), (255, 0, 0), 2  )
       cv2.putText(the_mask,'Mask', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
           
   return originalImage

# ===========
# for image
# ===========
mask_img = cv2.imread('7.jpg')
result =  detect(mask_img)
cv2.imshow('mask', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

