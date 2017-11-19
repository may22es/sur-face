import camera
from mainQApp import *
# from PySide.QtGui import *
# from PySide.QtCore import *

import sys


if __name__ == '__main__':
    if 1:   # QT app 사용시 1, 기존 cv만 사용시 0
        app = QApplication(sys.argv)
        # Create and show the form
        form = MyMainWindow()
        form.show()
        # Run the main Qt loop
        sys.exit(app.exec_())
    else:
        cam = camera.Camera()
        cam.run()

