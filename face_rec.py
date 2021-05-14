import face_recognition
import cv2 as cv
import os
from PIL import Image, ImageDraw
vidio_capture = cv.VideoCapture(0)



doni_image = face_recognition.load_image_file("doni.jpg")
doni_face_encoding = face_recognition.face_encodings(doni_image)[0]

ramsey_image = face_recognition.load_image_file("ramsey.jpg")
ramsey_face_encoding = face_recognition.face_encodings(ramsey_image)[0]

obama_image = face_recognition.load_image_file("obama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

biden_image = face_recognition.load_image_file("biden.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]



known_face_encodings = [
   
    ramsey_face_encoding,
    doni_face_encoding,
    obama_face_encoding,
    biden_face_encoding
]
known_face_names = [
    "ramsey",
    "doni",
    "obama",
    "biden"
]



while True:

    ret, frame = vidio_capture.read()
    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        
        matches =face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unkown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]




        cv.rectangle(frame, (left, bottom - 35), (right, bottom), (0,0,255), cv.FILLED)
        font = cv.FONT_HERSHEY_DUPLEX
        cv.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255,255,255), 1)


    cv.imshow('Vidio', frame)


    if cv.waitKey(1) & 0xFF == ord('q'):
        break

vidio_capture.release()
cv.destroyAllWindows()
print("Complete")