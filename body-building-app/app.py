import tkinter as tk
import customtkinter as ck

import numpy as np
import pandas as pd
import pickle

import mediapipe as mp
import cv2
from PIL import Image, ImageTk

import landmarks as landmarks

window = tk.Tk()
window.geometry("450x700")
window.title("Body Building App")
ck.set_appearance_mode("system")

classLabel = ck.CTkLabel(window, height=40, width=120, text="STAGE",font=("Arial", 20) ,text_color="black", padx=10)
classLabel.place(x=10, y=1)


counterLabel = ck.CTkLabel(window, height=40, width=120, text="REPS",font=("Arial", 20) , text_color="black", padx=10)
counterLabel.place(x=160, y=1)


probLabel = ck.CTkLabel(window, height=40, width=120, text="PROB",font=("Arial", 20) , text_color="black", padx=10)
probLabel.place(x=300, y=1)


classBox = ck.CTkLabel(window, height=40, width=120, text="0",font=("Arial", 20) , text_color="white", fg_color="blue")
classBox.place(x=10, y=41)


counterBox = ck.CTkLabel(window, height=40, width=120, text="0",font=("Arial", 20) , text_color="white", fg_color="blue")
counterBox.place(x=160, y=41)


probBox = ck.CTkLabel(window, height=40, width=120, text="0",font=("Arial", 20) , text_color="white", fg_color="blue")
probBox.place(x=300, y=41)


def reset_counter():
  global counter
  counter = 0

button = ck.CTkButton(window, text="RESET", command=reset_counter,  height=40, width=120,font=("Arial", 20) , text_color="white", fg_color="blue")
button.place(x=10, y=600)


frame = tk.Frame(height=400, width=400)
frame.place(x=10, y=90)
lmain = tk.Label(frame)
lmain.place(x=0, y=0)


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


with open('body-building-app/deadlift.pkl', 'rb') as f:
    model = pickle.load(f)


cap = cv2.VideoCapture(3)
current_stage = ''
counter = 0
bodylang_prob = np.array([0, 0])
bodylang_class = ''


def detect():
  global current_stage
  global counter
  global bodylang_prob
  global bodylang_class

  ret, frame = cap.read()
  image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  results = pose.process(image)
  mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                            mp_drawing.DrawingSpec(color=(100,13,173), thickness=4, circle_radius=5),
                            mp_drawing.DrawingSpec(color=(255,102,0), thickness=5, circle_radius=10))
   
  try:
    row = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
    X = pd.DataFrame([row], columns=landmarks)
    bodylang_class = model.predict_proba(X)[0]
    bodylang_prob = model.predict(X)[0]

    if bodylang_prob == 'down' and bodylang_prob[bodylang_prob.argmax()] > 0.7:
        current_stage = 'down'
    elif current_stage == 'down' and bodylang_prob == 'up' and bodylang_prob[bodylang_prob.argmax()] > 0.7:
      current_stage = 'up'
      counter = counter + 1
  except Exception as e:
    print(e)


  img = image[:, :400, :]
  imgarr = Image.fromarray(img)
  imgtk = ImageTk.PhotoImage(imgarr)  
  lmain.imgtk = imgtk
  lmain.configure(image=imgtk)
  lmain.after(10, detect)


  counterBox.configure(text=counter)
  probBox.configure(text=bodylang_prob[bodylang_prob.argmax()])
  classBox.configure(text=current_stage)


detect()


window.mainloop()
