import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import uic
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uic.loadUi("C://Users//HP//Desktop//mainwindow.ui", self)
        self.label_1 = QLabel("Model 1: ", self)# creating a label widget
        self.label_1.move(70, 520)# moving position
        #self.pushButton.setText("tung")


app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec_()