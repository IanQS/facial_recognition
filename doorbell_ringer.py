"""
Author: Caroline Dunn: https://github.com/carolinedunn/facial_recognition

Modified by I.Q
"""
import aiohttp
import asyncio
from imutils.video import VideoStream
import face_recognition
import imutils
import pickle
import time
import cv2
from datetime import datetime

import os
from twilio.rest import Client as TwilioClient
import base64
import requests


TWILIO_ACCT_SID = "ACd8b426370b26f5af5bcd0cf65b1d51"
TWILIO_AUTH_TOKEN = "873d4d22093352dca30bddb5e71bc8"
IMGBB_KEY = "e47cf08090dd72dcee4347291ecea1"
DEBUG = True
TWILIO_PHONE_NUMBER_TO_SEND_FROM = "+14439607012"
USER_TO_SEND_TO = ""
currentname = "unknown"

########################################
# Set up the messaging API
########################################

session = aiohttp.ClientSession()
twilio_client = TwilioClient(TWILIO_ACCT_SID, TWILIO_AUTH_TOKEN)

########################################
# Load the detectors
########################################
ENCODING_FNAME = "encodings.pickle"
CASCADE_XML = "haarcascade_frontalface_default.xml"

authorized_faces_encodings = pickle.loads(open(ENCODING_FNAME, "rb").read())
detector = cv2.CascadeClassifier(CASCADE_XML)

########################################
# Start up the "camera". In prod it would be a ring or some webcam
# attached to the door.
########################################
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

def send_mms(img, time):
    msg = "" #f"{time}_unidentified.png"
    
    cv2.imwrite(msg, img)
    with open(msg, "rb") as f:
        url = "https://api.imgbb.com/1/upload"
        payload = {
            "key": IMGBB_KEY,
            "image": base64.b64encode(f.read()),
        }
        res = requests.post(url, payload).json()
    uploaded_img_url = res["data"]["url"]
    if DEBUG:
        print(f"[DEBUG] URL to: {uploaded_img_url}")
    twilio_client.messages.create(
        to=USER_TO_SEND_TO,
        from_=TWILIO_PHONE_NUMBER_TO_SEND_FROM,
        body=msg,
        media_url=[uploaded_img_url]
    )
    os.remove(msg)

def get_frame():
    """
    Read the frame after our "doorbell" has been rung.

    Return the
        encoded faces as per https://face-recognition.readthedocs.io/en/latest/face_recognition.html
        doorbell_ringer_img: the image of the doorbell ringer(s)
        face_bounding_boxes: mostly for debugging (hence hte flag above)
    :return:
    """
    frame = vs.read()
    doorbell_ringer_img = imutils.resize(frame, width=500)
    time = datetime.now()
    fmt_time = f"{time.hour}:{time.minute} {time.day}-{time.month}"

    ########################################
    # First, convert into greyscale and then detect if there face(s)
    # present in the image. Then, pass these "face(s)" (if any) to our
    # classifier
    ########################################
    _grayscaled_doorbell_ringer_img = cv2.cvtColor(doorbell_ringer_img, cv2.COLOR_BGR2GRAY)

    # Need to get the face boxes within which our classifier will run
    face_bounding_boxes = detector.detectMultiScale(
        _grayscaled_doorbell_ringer_img, scaleFactor=1.1,
        minNeighbors=5, minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE)
    face_bounding_boxes = [(y, x + w, y + h, x) for (x, y, w, h) in face_bounding_boxes]

    ########################################
    # Take in the image of the doorbell ringer, and the bounded boxes
    # and produce the face embeddings
    ########################################
    rgb = cv2.cvtColor(doorbell_ringer_img, cv2.COLOR_BGR2RGB)
    # compute the facial embeddings for each face bounding box
    doorbell_faces_encodings = face_recognition.face_encodings(rgb, face_bounding_boxes)
    return doorbell_faces_encodings, doorbell_ringer_img, face_bounding_boxes, fmt_time


def identify_people(doorbell_faces_encodings, authorized_faces_encodings, img, fmt_time):
    """
    Identify the people in the image. If
    :param doorbell_faces_encodings:
        our detected faces encodings
    :param authorized_faces_encodings:
        our trusted faces encodings aka who can automatically come in
    :return:
    """
    names = []
    for single_face_encoding in doorbell_faces_encodings:
        matches = face_recognition.compare_faces(authorized_faces_encodings["encodings"],
                                                 single_face_encoding)
        name = "Unknown"  # if face is not recognized, then print Unknown

        if True in matches:
            print("[INFO] MATCHES EXIST")
            # find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                name = authorized_faces_encodings["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # determine the recognized face with the largest number
            # of votes (note: in the event of an unlikely tie Python
            # will select first entry in the dictionary)
            name = max(counts, key=counts.get)
        names.append(name)
    if DEBUG and names:
        print(f"[DEBUG] Recognized Users: {str(names)}")
    if not names:  # No people found
        if DEBUG:
            print(f"[DEBUG] No recognized faces. Sending MMS")
        send_mms(img, fmt_time)
    return names

def debug_faces(boxes, names, frame):

    # loop over the recognized faces
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # draw the predicted face name on the image - color is in BGR
        cv2.rectangle(frame, (left, top), (right, bottom),
                      (0, 255, 225), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                    .8, (0, 255, 255), 2)

def display_additional_info(names_list):
    for name in names_list:
        with open(f"additional/{name}.txt", "r") as f:
            data = f.readlines()

        print("*" * 10)
        for ln in data:
            print(data)
        print("*" * 10)


while True:
    run_ringer = input("Press R to simulate doorbell ring\n>>>")
    if run_ringer != "R":
        continue
    doorbell_faces_encodings, doorbell_ringer_img, face_boxes, fmt_time = get_frame()

    names = identify_people(doorbell_faces_encodings, authorized_faces_encodings, doorbell_ringer_img, fmt_time)
    
    if DEBUG:
        if not names:  # Only display in this case
            debug_faces(face_boxes, names, doorbell_ringer_img)
            cv2.imshow("Debugger", doorbell_ringer_img)
            cv2.waitKey(5000)

    display_additional_info(names)
