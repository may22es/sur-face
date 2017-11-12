import face_recognition
import cv2
import os

# Get a reference to webcam #0 (the default one)
cap = cv2.VideoCapture(0)

image = []
face_encoding = []
known_faces = []
faces_name = []
name = ''
filenames = os.listdir("./images/")
frame_color_known = (255, 255, 255)
frame_color_unknown = (0, 0, 255)
frame_color_display = frame_color_known

for idx in range(len(filenames)):
    image.append(face_recognition.load_image_file("./images/" + filenames[idx]))
    face_encoding.append(face_recognition.face_encodings(image[idx], num_jitters=100)[0])
    known_faces.append(face_encoding[idx])
    faces_name.append(filenames[idx][:-4]) #파일 이름에서 확장자 제거

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
scale = 2

while True:
    # Grab a single frame of video
    ret, frame = cap.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=1/scale, fy=1/scale)

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.4)

            # print(match)

            try :
                name = faces_name[match.index(True)]
            except ValueError as ve:
                name = 'Unknown'
            finally :
                # print(name)
                i = 1
            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= scale
        right *= scale
        bottom *= scale
        left *= scale

        # 얼굴인
        cv2.rectangle(frame, (left, top), (right, bottom), frame_color_display, 1)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 19), (right, bottom), frame_color_display, cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (0, 0, 0), 1)

    # Display the resulting image
    cv2.imshow('Video', cv2.resize(frame, (0, 0), fx=2, fy=2))

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
cap.release()
cv2.destroyAllWindows()
