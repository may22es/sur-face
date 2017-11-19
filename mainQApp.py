from ui_mainwindow import Ui_MainWindow
from mainCamera import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

import cv2


class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        self.setupUi(self)

        self.label_pos_view.setPixmap(QPixmap("resource/pos.jpg"))
        self.label_detect_view.setPixmap(QPixmap("resource/noone.jpg").scaled(100, 100))
        self.camera = MainCamera()
        self.camera.PictureSignal.connect(self.display_pic)
        # self.camera.PictureSignal.connect(self.face_detect)
        self.camera.DetectSignal.connect(self.display_detect)
        self.camera.CameraStateSignal.connect(self.ready_camera)
        self.camera.start()

    def closeEvent(self, *args, **kwargs):
        print("closed")
        self.camera.stop_thread()

    def ready_camera(self, status):
        if status:
            self.makePic.setEnabled()
            return
        self.makePic.setDisabled()

    def display_pic(self, picArray):
        pixmap = self.__array_to_QPixmap(picArray)
        pixmap = pixmap.scaled(self.label_camera_view.size())
        self.label_camera_view.setPixmap(pixmap)

    def __array_to_QPixmap(self, frame):
        picture = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # convert numpy mat to pixmap image
        qimg = QImage(picture.data, picture.shape[1], picture.shape[0], QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def display_detect(self, pic_name):
        # pixmap = self.__array_to_QPixmap(picArray)
        if pic_name == "Unknown":
            self.widget_unknown_input.setEnabled(True)
            self.labe_info.setText("")
            self.label_detect_view.setPixmap(QPixmap("resource/portrait.jpg").scaled(100, 100))
        elif pic_name == "NoOne":
            self.widget_unknown_input.setEnabled(False)
            self.labe_info.setText("")
            self.label_detect_view.setPixmap(QPixmap("resource/noone.jpg").scaled(100, 100))
        else:
            self.widget_unknown_input.setEnabled(False)
            self.labe_info.setText("회원 ID : 12345\n\n지난 번 주문 : 아메리카노\n\n오늘의 맞춤 추천 메뉴 : 레몬티\n\n")
            pixmap = QPixmap("images/" + pic_name + ".jpg").scaled(100, 100)
            self.label_detect_view.setPixmap(pixmap)



