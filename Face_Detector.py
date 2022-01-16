# importing the cv2 library to use the Haar Cascade algorithm to detect frontal faces and other common objects.
from tkinter import CASCADE
import cv2

# load pre-trained data on face frontals from opencv (Haar Cascade algorithm)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# choose an image file to detect a face within
# this reads the image into a 2D array
# img = cv2.imread('Elon_Musk.jpg')

# capture video from webcam
webcam = cv2.VideoCapture(0)

# Iterate forever over the frames of the video/stream
while True:
    
    # read the current frame
    successful_frame_read, frame = webcam.read()
    
    # must convert the raw image to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces of different sizes in the input image. Detected object will be returned as a list of rectangles
    face_coordinates = face_cascade.detectMultiScale(grayscaled_img, scaleFactor=1.0485258, minNeighbors=6, flags=0)

    # Draw the rectangle indicator around the detected face
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # print the face coordinates to terminal
    print(face_coordinates)

    # show the raw image
    cv2.imshow('Face Detector - Press Q to quit', frame)

    #  listen for a key press for 1 millisecond, then move on
    key = cv2.waitKey(1)

    # Stop if the Q key is pressed
    if key==81 or key == 113:
        break

# release the webcam
webcam.release()

print("\nProgram has Completed!\n")