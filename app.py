import streamlit as st
from streamlit_webrtc import webrtc_streamer,VideoProcessorBase,ClientSettings,WebRtcMode
import pickle
import cv2
import mediapipe as mp
import numpy as np
import time

WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
)
# Load the model
model_dict = pickle.load(open('model1.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'C',3:'D', 4:'E',5:'F',6:'G',7:'H',8:'I',9:'K',10:'L',
               11:'M',12:'N',13:'O',14:'P',15:'R',16:'S',17:'T',18:'U',19:'V',20:'W',21:'X',
               22:'Y',23:'Z'}

# Streamlit app
st.title("Sign Language Detection")
st.image('asl.webp',caption='American Sign Language')
st.markdown('NOTE: Use single hand to use ASL gesture')

run = st.button('Run')
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        print("Frame received") #dafasfdf
        img = frame.to_ndarray(format="bgr24")
        data_aux = []
        x_ = []
        y_ = []

        H, W, _ = img.shape

        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            prediction = model.predict([np.asarray(data_aux)])

            predicted_character = labels_dict[int(prediction[0])]

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(img, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

        return img

if run:
    time.sleep(1) # adding delay
    webrtc_streamer(key="example",mode=WebRtcMode.SENDRECV, client_settings=WEBRTC_CLIENT_SETTINGS, video_processor_factory=VideoProcessor)

# FRAME_WINDOW = st.image([])

# cap = cv2.VideoCapture(0)

# while run:
#     data_aux = []
#     x_ = []
#     y_ = []

#     ret, frame = cap.read()
#     if not ret:
#         break

#     H, W, _ = frame.shape

#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     results = hands.process(frame_rgb)
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 frame,  # image to draw
#                 hand_landmarks,  # model output
#                 mp_hands.HAND_CONNECTIONS,  # hand connections
#                 mp_drawing_styles.get_default_hand_landmarks_style(),
#                 mp_drawing_styles.get_default_hand_connections_style())

#         for hand_landmarks in results.multi_hand_landmarks:
#             for i in range(len(hand_landmarks.landmark)):
#                 x = hand_landmarks.landmark[i].x
#                 y = hand_landmarks.landmark[i].y

#                 x_.append(x)
#                 y_.append(y)

#             for i in range(len(hand_landmarks.landmark)):
#                 x = hand_landmarks.landmark[i].x
#                 y = hand_landmarks.landmark[i].y
#                 data_aux.append(x - min(x_))
#                 data_aux.append(y - min(y_))

#         x1 = int(min(x_) * W) - 10
#         y1 = int(min(y_) * H) - 10

#         x2 = int(max(x_) * W) - 10
#         y2 = int(max(y_) * H) - 10

#         prediction = model.predict([np.asarray(data_aux)])

#         predicted_character = labels_dict[int(prediction[0])]

#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
#         cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
#                     cv2.LINE_AA)

#     FRAME_WINDOW.image(frame)

# cap.release()
# #cv2.destroyAllWindows()
