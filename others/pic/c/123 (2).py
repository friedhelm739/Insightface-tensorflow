# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '123.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QAction

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(150, 110, 113, 20))
        self.lineEdit.setObjectName("lineEdit")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 23))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        MainWindow.setMenuBar(self.menubar)
        self.actionadd = QtWidgets.QAction(MainWindow)
        self.actionadd.setObjectName("actionadd")
        self.actionkeke = QtWidgets.QAction(MainWindow)
        self.actionkeke.setObjectName("actionkeke")
        self.menu.addAction(self.actionadd)
        self.menu.addAction(self.actionkeke)
        self.menubar.addAction(self.menu.menuAction())
        self.actionadd.triggered.connect(self.add) 
        self.actionkeke.triggered.connect(self.keke) 
        
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.menu.setTitle(_translate("MainWindow", "文件"))
        self.actionadd.setText(_translate("MainWindow", "Add New"))
        self.actionkeke.setText(_translate("MainWindow", "Update"))

    def add(self):
        self.lineEdit.setText("actionadd")
        
    def keke(self):        
        self.lineEdit.setText("actionkeke")


import sys
from PyQt5.QtWidgets import QApplication, QMainWindow


class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        self.setupUi(self)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MyMainWindow()
    myWin.show()
    sys.exit(app.exec_())          
