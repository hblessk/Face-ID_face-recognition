import sys
import cv2
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic
import time
from PyQt5.QtCore import QStringListModel
import pandas as pd
from scipy.io import mmread
import pickle
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import linear_kernel
import re
from konlpy.tag import Okt


form_window = uic.loadUiType('./unlocked_video_app.ui')[0]
# class Exam(QWidget, form_window):
#     def __init__(self):
#         super().__init__()
#         self.setupUi(self)

class PThread(QThread):
    changePixmap = pyqtSignal(QImage)

    def __init__(self, mainWindow):  # mainWindow는 QWidget에서 전달하는 self이다.(QWidget의 인스턴스)
        super().__init__(mainWindow)
        self.mainWindow = mainWindow  # self.mainWindow를 사용하여 QWidget 위젯을 제어할 수 있다.
        print('debug 1')

    def run(self):  # 쓰레드로 동작시킬 함수 내용 구현(런에 원하는 동작을 넣어줘야한다!)
        cap = cv2.VideoCapture(0)
        # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print('debug 2')

        while True:
            ret, img = cap.read()
            if ret:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                h, w, c = img.shape
                qimg = QImage(img.data, w, h, w*c, QImage.Format_RGB888)

                p = qimg.scaled(640, 640, Qt.KeepAspectRatio)

                # self.mainWindow.lbl_image.setPixmap(p)

                self.changePixmap.emit(p)

                time.sleep(1/30)

            else:
                QMessageBox.about(self, "Error", "Cannot read frame.")


# 쓰레드 클래스 만들어주고
class Exam(QWidget, form_window):
    def setImage(self, image):
        self.lbl_image.setPixmap(QPixmap.fromImage(image))
        print('debug 9')

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        # self.lbl_result.setText('exam')
        th = PThread(self)  # self는 WindowClass의 인스턴스, Thread 클래스에서 mainWindow로 전달
        th.changePixmap.connect(self.setImage)
        th.start()  # 쓰레드 클래스의 run 메서드를 동작시키는 부분(위젯이랑 쓰레드 연결(run))
        print('debug 8')


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = Exam()
    mainWindow.show()
    sys.exit(app.exec_())