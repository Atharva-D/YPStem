from __future__ import division, print_function
#import sys
import os
import cv2
#import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import statistics as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from scipy import stats
from urllib import response
import requests
import json
from urllib.request import urlopen

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index1.html")

@app.route('/camera', methods = ['GET', 'POST'])

def camera():
    mp_holistic = mp.solutions.holistic # Holistic model
    mp_drawing = mp.solutions.drawing_utils # Drawing utilities
    def mediapipe_detection(image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
        image.flags.writeable = False                  # Image is no longer writeable
        results = model.process(image)                 # Make prediction
        image.flags.writeable = True                   # Image is now writeable 
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
        return image, results

    def draw_landmarks(image, results):
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

    def draw_styled_landmarks(image, results):
        # Draw left hand connections
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                 ) 
        # Draw right hand connections  
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 ) 

    def extract_keypoints(results):
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([ lh, rh])

    actions = np.array(['accident', 'ambulance', 'breathe','emergency', 'fire', 'help'])

    colors = [(245,117,16), (117,245,16), (16,117,245),(255,0,0),(255,140,0),(0,0,0)]
    def prob_viz(res, actions, input_frame, colors):
        output_frame = input_frame.copy()
        for num, prob in enumerate(res):
            cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
            cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

        return output_frame
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.5
    model = load_model('C:/Users/Shreya Daga/Downloads/ActionDetectionforSignLanguage_website/ActionDetectionforSignLanguage_website/action.h5')

    cap = cv2.VideoCapture(0)
    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        i=0
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            print(results)

            # Draw landmarks
            draw_styled_landmarks(image, results)

            # 2. Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])
                predictions.append(np.argmax(res))


            #3. Viz logic
                if np.unique(predictions[-10:])[0]==np.argmax(res): 
                    if res[np.argmax(res)] > threshold: 

                        if len(sentence) > 0: 
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5: 
                    sentence = sentence[-5:]

                # Viz probabilities
                image = prob_viz(res, actions, image, colors)

            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Show to screen
            cv2.imshow('OpenCV Feed', image)
#            i=i+1
            

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                
                
        print(sentence)
        cap.release()
        cv2.destroyAllWindows()
        #final_output1 = st.mode(sentence)
        final_output1=sentence[-1]
        
        url='http://ipinfo.io/json'
        response = urlopen(url)
        data=json.load(response)
        
        url = "https://www.fast2sms.com/dev/bulk"
        message = "a"
        

        if final_output1 == "accident":
            message = "Accident!!!!!!!....... Pls Help "+data['ip']+","+data['city']+","+data['region']+","+data['loc']
        elif final_output1 == "ambulance":
            message = "Amulance required!!!!!!!....... Pls Help "+data['ip']+","+data['city']+","+data['region']+","+data['loc']
        elif final_output1 == "emergency":
            message = "Emergency!!!!!!!....... Pls Help "+data['ip']+","+data['city']+","+data['region']+","+data['loc']
        elif final_output1 == "breathe":
            message = "Relax, Breathe!!!!!!!....... Pls Help "+data['ip']+","+data['city']+","+data['region']+","+data['loc']
        elif final_output1 == "fire":
            message = "Fire Alert!!!!!!!....... Pls Help "+data['ip']+","+data['city']+","+data['region']+","+data['loc']
        elif final_output1 == "help":
            message = "Help Required!!!!!!!....... Pls Help "+data['ip']+","+data['city']+","+data['region']+","+data['loc']

        my_data ={
            'sender_id':"FSTSMS",
            'message':message,
            'language':'english',
            'route':'p',
            'numbers':'9370394253'
        }

        headers ={
            'authorization':'VOJYZzEsS1LnDCI5KFgT4ym8wHeAMqbQP9uXh6tRdk3Npocj2aA4jbQaVuzrhO1W9MpkTvCNRdfix257',
            'Content-Type':'application/x-www-form-urlencoded',
            'Cache-Control':'no-cache'
        }

        response = requests.request("POST",url,data = my_data,headers=headers)
        returned_msg = json.loads(response.text)

        print(returned_msg['message'])
    
        return render_template("buttons.html",final_output=final_output1)
       

if __name__ == "__main__":
    app.run()