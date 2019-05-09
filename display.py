# -*- coding: utf-8 -*-
"""
Created on Wed May  8 21:19:26 2019

@author: friedhelm
"""
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import *
from core.structure import Ui_Form
import sys
import cv2
import os
from core import config
from recognizer.arcface_recognizer import Arcface_recognizer

class MyMainWindow(QMainWindow):
    
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        self.Ui_Form=Ui_Form()
        self.Ui_Form.setupUi(self)
        self.timer_camera = QtCore.QTimer()   
        self.timer_camera.stop()    
        self.cap = cv2.VideoCapture()
        self.timer_camera.timeout.connect(self.vedio_show)
        self.recognize_flag = False
        self.recognizer = Arcface_recognizer(config.arc_model_name, config.arc_model_path, config.mtcnn_model_path)
        
    def vedio_show(self):
        flag, self.image = self.cap.read()

        img = cv2.resize(self.image, (640, 480))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if(self.recognize_flag):
            
            names, bounding_boxes = self.recognizer.recognize(img)
            
            if(len(names)!=0):
                for idx, name in enumerate(names):
                    if name is not None:
                        cv2.rectangle(img, (bounding_boxes[idx][0],bounding_boxes[idx][1]), (bounding_boxes[idx][2], bounding_boxes[idx][3]), (255, 255, 255), thickness=2)
                        cv2.putText(img, str(name), (bounding_boxes[idx][0],bounding_boxes[idx][1]), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0),thickness=2)


        showImage = QtGui.QImage(img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_RGB888)
        self.Ui_Form.label.setPixmap(QtGui.QPixmap.fromImage(showImage))


    def OpenVedio(self):      
        if self.timer_camera.isActive() == False:
            flag = self.cap.open(0)
            if flag == False:
                self.Ui_Form.msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"请检测相机与电脑是否连接正确", buttons=QtWidgets.QMessageBox.Ok,
                                                defaultButton=QtWidgets.QMessageBox.Ok)
                  
            else:
                self.timer_camera.start(30)
                self.Ui_Form.pushButton.setText("关闭摄像头")

        else:
            self.timer_camera.stop()
            self.cap.release()
            self.Ui_Form.label.clear()
            self.Ui_Form.pushButton.setText("打开摄像头")

    def StartRecognize(self): 
        
        if not self.recognize_flag:
            self.Ui_Form.pushButton_2.setText("停止识别")
        else:
            self.Ui_Form.pushButton_2.setText("开始识别")
        
        self.recognize_flag = not self.recognize_flag
        
        return 


    def AddNew(self): 
        
        addr = self.Ui_Form.lineEdit.text()
        
        if not os.path.exists(addr):
            self.Ui_Form.lineEdit.setText("请输入正确的地址")
        else:    
            self.recognizer.add_customs(addr)
            self.Ui_Form.lineEdit.setText("完成！")
        
        return


    def closeEvent(self, event):
        self.Ui_Form.ok = QtWidgets.QPushButton()
        self.Ui_Form.cacel = QtWidgets.QPushButton()
 
        self.Ui_Form.msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, u"关闭", u"是否关闭！")
 
        self.Ui_Form.msg.addButton(self.Ui_Form.ok,QtWidgets.QMessageBox.ActionRole)
        self.Ui_Form.msg.addButton(self.Ui_Form.cacel, QtWidgets.QMessageBox.RejectRole)
        self.Ui_Form.ok.setText(u'确定')
        self.Ui_Form.cacel.setText(u'取消')

        if self.Ui_Form.msg.exec_() == QtWidgets.QMessageBox.RejectRole:
            event.ignore()
        else:
            if self.cap.isOpened():
                self.cap.release()
            if self.timer_camera.isActive():
                self.timer_camera.stop()
            self.recognizer.close_db()
            event.accept()
        return


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MyMainWindow()
    myWin.show()
    sys.exit(app.exec_())         
