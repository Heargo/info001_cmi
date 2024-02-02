import face_recognition
import cv2
import numpy as np
import os
import time
import ctypes
import tensorflow as tf
from PIL import Image
import keras
from keras.models import load_model


#import emotion model from hdf5 file
model = load_model('model_emotions/content/model.ckpt')
# import age model
model_age = keras.models.load_model('model_age/content/model.ckpt')


# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
frame_rate_processed = 5
process_this_frame = True
i = 0
last_time_safe = time.time()
emotions_classes = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]
emotions_colors = {
    "angry": (0, 0, 255), #red
    "disgust": (128, 195, 135), # puke green
    "fear": (237, 96, 237), # light blue
    "happy": (0, 255, 0), # green
    "neutral": ( 199, 199, 199 ), # grey
    "sad": (237, 201, 93), # blue
    "surprise": (49, 165, 240) # orange
}

def image_to_emotion_format(image):
    # Convert image to RGB (3 channels)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize image to 48x48
    image_resized = cv2.resize(image_rgb, (48, 48))
    # show the image
    cv2.imshow("image", image_resized)
    return image_resized

def sub_image(image, top, right, bottom, left):
    #numpy to pil image
    img = Image.fromarray(image)
    cropped = img.crop((left, top, right, bottom))
    #pil image to numpy
    cropped = np.array(cropped)
    return cropped

def predict_emotion(image):
    #convert image to emotion format
    image = image_to_emotion_format(image)
    #convert image to numpy array
    image = np.array(image)
    #reshape image to 1x48x48x1
    image = np.reshape(image,(48, 48, 3))
    #predict the emotion
    emotion = model.predict(tf.expand_dims(image, axis=0))
    #return the emotion
    return emotions_classes[np.argmax(emotion)]

def predict_age(image):
    #convert image to emotion format
    image = image_to_emotion_format(image)
    #convert image to numpy array
    image = np.array(image)
    #reshape image to 1x48x48x1
    image = np.reshape(image,(48, 48, 3))
    #predict the emotion
    age = model_age.predict(tf.expand_dims(image, axis=0))
    av=0
    for i in range(84):
        av+= i * age[0][i] 
    #return the emotion
    return av

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Only process every other frame of video to save time
    if process_this_frame:
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        print("face locations: ", face_locations)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_location in face_locations:
            # get face sub image
            top, right, bottom, left = face_location
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            face_image = sub_image(frame, top, right, bottom, left)
            # predict emotion
            emotion = predict_emotion(face_image)
            age = predict_age(face_image)
            face_names.append(emotion+" "+str(age)+" ans")

    i = (i+1) % frame_rate_processed
    process_this_frame = i==0


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        #green if name is safe red if not
        em = name.split(" ")[0]
        color = emotions_colors[em]
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # if the last time safe was more than 20 seconds ago then lock the screen (simulate windows + l keystroke)
    # if time.time() - last_time_safe > 10:
    #     print("lock the screen")
    #     ctypes.windll.user32.LockWorkStation()


# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()