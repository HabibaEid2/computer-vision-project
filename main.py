import cv2
import os 
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
imagesPath = os.listdir('users')
usersNames = []
facesList = []

for img in imagesPath : 
    usersNames.append(img[:img.index('.')])
    img = cv2.cvtColor(cv2.imread(f'users/{img}'), cv2.COLOR_BGR2GRAY)
    imgFaces = face_cascade.detectMultiScale(img , 1.3 , 5)
    (x , y , w , h) = imgFaces[0]
    face = img[y:y+h+50 , x:x+w]
    face = cv2.resize(face , (200 , 200))
    facesList.append(face) 

cap = cv2.VideoCapture(0)

while True : 
    ret , frame = cap.read()
    gray_frame = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    facesFounded = face_cascade.detectMultiScale(gray_frame , 1.3 , 5)

    for i in range(len(facesFounded)) : 
        (x , y , w , h) = facesFounded[i]

        face = gray_frame[y:y+h+50 , x:x+w]
        face = cv2.resize(face , (200 , 200))

        model = cv2.face.LBPHFaceRecognizer.create()
        model.train(facesList , np.array([0 , 1 , 2 , 3]))

        label , confidence = model.predict(face)

        if confidence < 60 : 
            cv2.putText(frame, usersNames[label], (x , y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0 , 255 , 0) , 1, cv2.LINE_AA)
            cv2.rectangle(frame , (x , y) , (x + w , y + h) , (0 , 255 , 0) , 2)
        else : 
            cv2.putText(frame, 'Unknown', (x , y- 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0 , 0 , 255) , 1, cv2.LINE_AA)
            cv2.rectangle(frame , (x , y) , (x + w , y + h) , (0 , 0 , 2500) , 2)

    cv2.imshow('my video' , frame)
    if cv2.waitKey(1) == ord('q') : 
        break 

cap.release()
cv2.destroyAllWindows()