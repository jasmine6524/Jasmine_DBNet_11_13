import os
import sys
import time
import numpy as np
import cv2
from PyQt5.QtCore import QTime, QTimer
from PyQt5.QtGui import QImage, QPixmap, QWindow

from PyQt5.QtWidgets import *
from draft_layout_12 import Ui_MainWindow
from PIL import Image, ImageQt

import speech_recognition as sr
import py
import pyttsx3 as pt
import mac_say
import pyttsx3 as pt
import time
import keyboard
from aip import AipOcr

class myWindow(QWidget, Ui_MainWindow):
    def __init__(self):
        super(myWindow, self).__init__()
        self.initUI()
        self.initArgs()
        self.initSlot()
        
        
    def initSlot(self):
        self.imageBtn.clicked.connect(self.selectImage)
        self.timer.timeout.connect(self.showFrame)
        self.open_camera.clicked.connect(self.openCamera)
        self.capture_button.clicked.connect(self.capture)
        self.close_camera.clicked.connect(self.closeCamera)
        # self.recogninze_btn.clicked.connect(self.inference)

    def initUI(self):
        self.win = QMainWindow()
        self.setupUi(self.win)

    def initArgs(self):
        self.timer = QTimer()
        self.cap = None
        self.flag  = False
        self.frame = None

    def openCamera(self):
        self.cap = cv2.VideoCapture(0)#0是表示调用电脑自带的摄像头，1是表示调用外接摄像头
        self.timer.start(100)

    def showFrame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (591, 332))
            self.frame = frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[1]*3, QImage.Format_RGB888)
            self.cameraArea.setPixmap(QPixmap.fromImage(frame))
    
    def inference(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite('./recog/test_imgs/tmp.jpg', img)
        img = cv2.imread('./recog/test_imgs/tmp.jpg')
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        print(img.shape)
        print(type(img))
        cv2.imshow('e',img)
        cv2.waitKey(10000)
        os.system('python inference.py')
        img = cv2.imread('./outputs_test/img_result/tmp.jpg')
        # print(img)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame = QImage(img.data, img.shape[1], img.shape[0], img.shape[1]*3, QImage.Format_RGB888)
        self.cameraArea.setPixmap(QPixmap.fromImage(frame))
        mac_say.say('Detection Completed')
        mac_say.say('Start recognizing text')
        os.system('python ./recog/tools/infer/inference.py --e2e_algorithm="PGNet" --image_dir="./recog/test_imgs/tmp.jpg" --e2e_model_dir="./recog/models" --e2e_pgnet_polygon=True --use_gpu=False')
        mac_say.say('Recognizing')
        text_path = './recog/inference_results/res_texts/e2e_res_tmp.txt'
        # recognized_words = []
        # with open(text_path, 'w') as f1:
        #     output = recognize_image('./outputs_test/img_result/tmp.jpg')

            
        #     for word in output['words_result']:
        #         for key,value in word.items():
        #             print(value)
        #             recognized_words.append(value + '\n')
            

        #     print(recognized_words)
        #     for i in recognized_words:
        #         f1.write(str(i) + '\n')
            
        
        with open(text_path, 'r', encoding='utf8') as f:
            lines = f.readlines()
            lines = [line.strip('\n') for line in lines]
            s = ''
            for line in lines: s += line + ' '
        self.resultArea.setText(s)
        mac_say.say('Recognition Completed, the text is ')
        mac_say.say(s)

    # def deleteImgs(self):
    #     os.remove('./recog/test_imgs/tmp.jpg')
    #     os.remove('./recog/inference_results/res_imgs/e2e_res_tmp.jpg')
    #     os.remove('./recog/inference_results/res_texts/e2e_res_tmp.txt')
    #     os.remove('./outputs_test/img_result/tmp.jpg')
    #     os.remove('./outputs_test/img_text/res_tmp.txt')

    def capture(self):
        self.flag = True
        self.closeCamera()
        mac_say.say('Start detecting text')
        self.inference(self.frame)
        # self.deleteImgs()
        self.flag = False

    def closeCamera(self):
        self.timer.stop()
        if not self.flag:
            self.cameraArea.clear()
            self.cap.release()
            # cv2.destroyAllWindows()


    def selectImage(self):
        '''
        function: 上传一张电脑上的图片
        '''
        imageName, imageType = QFileDialog.getOpenFileName(
            self,
            'select image',
            os.getcwd(),
            # 'Jpg Files(*.jpg);;Png Files(*.png)'
        )
        if imageName == '':
            msg = (QMessageBox.warning(self, 'Warning', 'Please select an image', QMessageBox.Ok, QMessageBox.Ok))
        else:
            self.imagePath.setText(imageName)
            self.showImage(imageName)
            # self.inference(self.frame)
    
    def showImage(self, path):
        '''
        function: 根据图片的大小进行相应的调整来将一个图片完整的现实在display框里
        parameters: path = 路径
        '''
        img = Image.open(path)
        w, h = img.size
        if w > 421 and h < 301: #421是display的宽度，301是display的高度
            r = w / 421
            img = img.resize((421, int(h/r)), 4)
        elif h > 301 and w < 481:
            r = h / 301
            img = img.resize((int(w/r), 281), 4)
        elif h > 301 and w > 481:
            r = h / 301
            img = img.resize((int(w/r), 281), 4)
        print(img)
        print(type(img))
        img.show()
        img = np.array(img)
        # img = img[:,:,1:3]
        print(img.shape)
        # img = np.transpose(img,(1,2,0))
        # img = ImageQt.toqpixmap(img)
        print(img.shape)
        self.inference(img)

        # img.show()
        # self.frame = self.cameraArea.setPixmap(img)
        print(self.frame)
    
    def show(self):
        self.win.show()

def voiceControl(win):
    r = sr.Recognizer()
    # print (sr.Microphone.list_microphone_names()[0])
    mic = sr.Microphone(device_index=0)
    #mac_say.say('1, say open to open the camera, 2, say close to close the camera, 3, press c to capture a frame')
    while True:
        #sp.say('input your command')
        mac_say.say("input your command")
        cmd = input()
        if cmd == 'r':
            with mic as source:
                r.adjust_for_ambient_noise(source)
                mac_say.say('ready to say your command')
                audio = r.listen(source)
                cmd = r.recognize_google(audio)
                print(cmd)
                if cmd == 'open':
                    mac_say.say('camera opened')
                    win.openCamera()
                elif cmd == 'close':
                    mac_say.say('camera closed')
                    win.closeCamera()   
                elif cmd == 'capture':
                    mac_say.say('frame captured')
                    win.capture()
        else:
            break 
       
def recognize_image(path):
    APP_ID = '24949285'
    API_KEY = 'u3Gf0GVIWXIKnuhZT849GUIA'
    SECRET_KEY = 'I98UVLt4HWzWfR1KjAMGxuwunEpOhoqx'

    client = AipOcr(APP_ID, API_KEY, SECRET_KEY)

    """ 读取图片 """
    def get_file_content(filePath):
        with open(filePath, 'rb') as fp:
            return fp.read()

    image = get_file_content(path)

    result = client.basicGeneral(image)
    return result
      


if __name__ == '__main__':
    app = QApplication(sys.argv) #application是底层, window在上面
    win = myWindow()
    win.show()
    voiceControl(win)
    app.exec_()
    sys.exit() #不断的重复