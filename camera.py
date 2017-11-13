import face_recognition
import cv2
import os
import pickle

ESC = 27


class Camera:
    image = []
    face_encoding = []
    known_faces = []
    faces_name = []
    name = ''
    file_names = os.listdir("./images/")
    frame_color_known = (255, 255, 255)
    frame_color_unknown = (0, 0, 255)
    frame_color_display = frame_color_known

    process_this_frame = True
    scale = 2

    def __init__(self, cam_idx=0):
        self.cap = cv2.VideoCapture(cam_idx)
        self.load_images()

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()

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
                        self.face_encoding.append(face_recognition.face_encoding(self.image[idx], num_jitters=100)[
                                                      0])  # num_jitters=1 is 100 times faster and worse than 100
                        self.known_faces.append(self.face_encoding[idx])

                    f.truncate()
                    pickle.dump({'faces_name': self.faces_name, 'known_faces': self.known_faces}, f)

    def run(self):
        while True:
            # Grab a single frame of video
            ret, frame = self.cap.read()

            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=1 / self.scale, fy=1 / self.scale)

            # Only process every other frame of video to save time
            if self.process_this_frame:
                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(small_frame)
                face_encodings = face_recognition.face_encodings(small_frame, face_locations)

                face_names = []
                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    match = face_recognition.compare_faces(self.known_faces, face_encoding, tolerance=0.4)

                    # print(match)

                    try:
                        name = self.faces_name[match.index(True)]
                    except ValueError as ve:
                        name = 'Unknown'
                    finally:
                        # print(name)
                        i = 1
                    face_names.append(name)

            self.process_this_frame = not self.process_this_frame

            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations
                top *= self.scale
                right *= self.scale
                bottom *= self.scale
                left *= self.scale

                # 얼굴인식
                cv2.rectangle(frame, (left, top), (right, bottom), self.frame_color_display, 1)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 19), (right, bottom), self.frame_color_display, cv2.FILLED)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (0, 0, 0), 1)

            # Display the resulting image
            if os.name == "posix":  # linux option
                frame = cv2.resize(frame, (0, 0), fx=2, fy=2)
            cv2.imshow('Video', frame)

            # Hit 'q' or 'ESC' on the keyboard to quit!
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == ESC:
                break
