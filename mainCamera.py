# from PySide.QtCore import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import face_recognition

import os
import pickle
import cv2
import numpy as np

from queue import Queue

class MainCamera(QThread):
    # PictureSignal = Signal(np.ndarray)
    PictureSignal = pyqtSignal(np.ndarray)
    # CameraStateSignal = Signal(bool)
    CameraStateSignal = pyqtSignal(bool)
    DetectSignal = pyqtSignal(str)

    cap = None

    image = []
    face_encoding = []
    known_faces = []
    faces_name = []
    name = ''
    file_names = os.listdir("./images/")
    frame_color_known = (255, 255, 255)
    frame_color_unknown = (0, 0, 255)
    frame_color_display = frame_color_known

    queue_size = 5   # maxsize of queue
    name_queue = Queue(queue_size)

    process_this_frame = True
    scale = 4

    def __init__(self, *args, **kwargs):
        QThread.__init__(self, *args, **kwargs)
        self.camRun = True
        for _ in range(self.queue_size):
            self.name_queue.put_nowait("NoOne")
        self.frame_cnt = 0
        self.load_images()
        try:
            self.cap = cv2.VideoCapture(0)
            self.CameraStateSignal.emit(True)
        except Exception:
            self.CameraStateSignal.emit(False)
            pass

    def run(self):
        while self.camRun:
            self.capture_pic()
            self.frame_cnt = self.frame_cnt+1

    def capture_pic(self):
        ret, frame = self.cap.read()
        self.detect([frame])
        self.PictureSignal.emit(frame)

    def detect(self, frame_):
        frame = frame_[0]
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=1 / self.scale, fy=1 / self.scale)
        # Only process every other frame of video to save time
        if self.process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(small_frame)
            for (top, right, bottom, left) in face_locations:
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= self.scale
                right *= self.scale
                bottom *= self.scale
                left *= self.scale

                cv2.rectangle(frame, (left, top), (right, bottom), self.frame_color_display, 2)

            if self.frame_cnt&10 == 0:  # 10 frame 마다 compare_faces
                face_encodings = face_recognition.face_encodings(small_frame, face_locations)

                face_names = []
                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    match = face_recognition.compare_faces(self.known_faces, face_encoding, tolerance=0.4)

                    try:
                        name = self.faces_name[match.index(True)]
                    except ValueError as ve:
                        name = 'Unknown'
                    finally:
                        pass
                    face_names.append(name)
                if face_names:
                    self.name_queue.get_nowait()
                    self.name_queue.put_nowait(face_names[0])
                    # self.DetectSignal.emit(face_names[0])
                else:
                    self.name_queue.get_nowait()
                    self.name_queue.put_nowait("NoOne")

                name_list = list(self.name_queue.queue)
                if name_list.count("Unknown") > self.queue_size*0.7:
                    self.DetectSignal.emit("Unknown")
                elif name_list.count("NoOne") > self.queue_size*0.7:
                    self.DetectSignal.emit("NoOne")
                else:
                    name_list = list(filter(("Unknown").__ne__, name_list))
                    name_list = list(filter(("NoOne").__ne__, name_list))
                    if name_list:
                        name = max(set(name_list), key=name_list.count)
                        self.DetectSignal.emit(name)

    def stop_thread(self):
        self.camRun = False

    def load_images(self):
        for idx in range(len(self.file_names)):
            self.faces_name.append(self.file_names[idx][:-4])

        with open('encoding.bin', (lambda: "rb+" if os.path.isfile('encoding.bin') else "wb+")()) as f:     # binary read and write
            if os.stat("encoding.bin").st_size == 0:   # if encoding.bin is empty
                print("create encoding.bin")
                for idx in range(len(self.file_names)):
                    self.image.append(face_recognition.load_image_file("./images/" + self.file_names[idx]))
                    self.face_encoding.append(face_recognition.face_encodings(self.image[idx], num_jitters=100)[0])
                    self.known_faces.append(self.face_encoding[idx])

                pickle.dump({'faces_name': self.faces_name, 'known_faces': self.known_faces}, f)
            else:
                print("compare with encoding.bin")
                p = pickle.load(f)

                # 현재 jpeg file 이름과 encoding에 저장된 file 이름을 비교
                if self.faces_name == p['faces_name']:
                    print("same with encoding.bin")
                    self.known_faces = p['known_faces']
                else:
                    print("different with encoding.bin")
                    for idx in range(len(self.file_names)):
                        self.image.append(face_recognition.load_image_file("./images/" + self.file_names[idx]))
                        self.face_encoding.append(face_recognition.face_encodings(self.image[idx], num_jitters=100)[0])
                        self.known_faces.append(self.face_encoding[idx])

                    f.truncate()
                    pickle.dump({'faces_name': self.faces_name, 'known_faces': self.known_faces}, f)
