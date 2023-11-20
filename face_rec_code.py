##check below how to view the image##

import cv2
import dlib
cap = cv2.VideoCapture(1)  # Use default camera
image_counter = 0 
predictor = dlib.shape_predictor("/Users/niharpalem/Desktop/masters docs/fall-23/DATA-270/face rec/python codes/shape_predictor_68_face_landmarks.dat")  # You need to download this file
 # Counter for the image name
detector = dlib.get_frontal_face_detector()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = detector(gray, 1)

    # Get the landmarks/parts for the face in box d.
    for i, d in enumerate(faces):
        shape = predictor(gray, d)
        for i in range(1, 68):  # There are 68 landmark points on each face
            cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 3, (0, 255, 0), -1)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):  # Exit the loop when 'q' is pressed
        break
    if key == ord('p'):  # Take a picture when 'p' is pressed
        img_name = "opencv_frame_{}.png".format(image_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        image_counter += 1

cap.release()
cv2.destroyAllWindows()
 

##to view the image##
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

for i in range(image_counter):
    img_name = "opencv_frame_{}.png".format(i)
    img = mpimg.imread(img_name)
    imgplot = plt.imshow(img)
    plt.show()
