import face_recognition
import cv2
import numpy as np
import os
import time
import ctypes
import tensorflow as tf
from PIL import Image


# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
frame_rate_processed = 20
process_this_frame = True
i = 0
last_time_safe = time.time()

orginal_dataset = "./old_dataset/all"
dataset_age = "./dataset_age"
dataset_genre = "./dataset_genre"
# duplicate folder structure from orginal_dataset to new_dataset
for age in range(1,110):
    if not os.path.exists(dataset_age+"/"+str(age)):
        os.makedirs(dataset_age+"/"+str(age))

if not os.path.exists(dataset_genre): 
    os.makedirs(dataset_genre)
    if(not os.path.exists(dataset_genre+"/male")):
        os.makedirs(dataset_genre+"/male")
    if(not os.path.exists(dataset_genre+"/female")):
        os.makedirs(dataset_genre+"/female")

def save_image(image,path):
    #get the current time
    current_time = time.time()
    #format:
    img = image_to_new_format(image)
    #convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    save_path = path+"/"+str(current_time)+".jpg"
    print("saving image to: ", save_path)
    #save the image
    cv2.imwrite(save_path, img)

def image_to_new_format(image):
    # Resize image to 48x48
    image_resized = cv2.resize(image, (48, 48))
    return image_resized

def sub_image(image, top, right, bottom, left):
    #numpy to pil image
    img = Image.fromarray(image)
    cropped = img.crop((left, top, right, bottom))
    #pil image to numpy
    cropped = np.array(cropped)
    return cropped



#for each file in videos folder
for file in os.listdir(orginal_dataset):
    try:
        #get age (first number of the file name)
        age = int(file.split("_")[0])
        gender = int(file.split("_")[1])
        if(file.split("_")[0] =="9"):
            print("processing file: ", file)
            #get the image
            frame = cv2.imread(orginal_dataset+"/"+file)
            # Resize frame  to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
            # Find all the faces and face encodings in the current frame
            face_locations = face_recognition.face_locations(rgb_small_frame)
            print("face locations: ", face_locations)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            for face_location in face_locations:
                # get face sub image
                top, right, bottom, left = face_location
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                face_image = sub_image(frame, top, right, bottom, left)
                #save the image in the age dataset
                save_image(face_image, dataset_age+"/"+str(age))
                #save the image in the genre dataset
                gender_folder = "male" if gender == 0 else "female"
                save_image(face_image,dataset_genre+"/"+str(gender_folder))
    except Exception as e:
        print("error on file",file,": ", e)



       