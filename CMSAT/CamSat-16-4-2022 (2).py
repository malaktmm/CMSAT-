import os
import sys
from PyQt5.QtWidgets import QMainWindow, QAction, qApp, QApplication
from PyQt5.QtGui import QIcon
from PyQt5.uic import loadUi
from PyQt5 import uic
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QPixmap, QImage
import subprocess
import cv2
import numpy as np           

from cgitb import enable
import sys
import os

from PIL import ImageTk,Image  
import imutils
import argparse
import multiprocessing
import imageio
#import tensorflow as tf

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.svm import SVC

import requests
import json


qtCreatorFile = "design.ui"  # Enter file here.
global ImageFile
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)

# for video capturing
e = multiprocessing.Event()
p = None
    
#To start use HOG technique ------------------------------------------------
HOGCV = cv2.HOGDescriptor()
HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())



class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent=parent)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.comboBox.addItem("Image body")
        self.comboBox.addItem("Image face")
        self.comboBox.addItem("Camera")
        self.comboBox.addItem("Video")
        self.Btn_Browse.clicked.connect(self.Browser)
        self.Btn_Classifier.clicked.connect(self.classifier)
        self.Btn_Start.clicked.connect(self.startDetect)
        self.Btn_getData.clicked.connect(self.getWebData)
        
        '''
        self.browse.clicked.connect(self.Test)
        self.classifierBtn.clicked.connect(self.runClassifier)
        self.showNext.clicked.connect(self.moveNext)
        self.close.clicked.connect(self.Close)
        self.comboBox.addItem("First Disease")
        self.comboBox.addItem("Second Disease")
        '''

    # database requests
    def getWebData(self):
        webDataInfos = [] # for storeing the descisions
        url = "http://cmsat.atwebpages.com/scripts.php"
        myobj = {'act':'getData', }
        x = requests.post(url, data = myobj)
        results = json.loads(x.text)
        print(results)
        textvalue = ''
        for res in results:
            textvalue += str(res["username"]) + " - " + str(res["mtype"]) + " = " + str(res["mvalue"]) + "\n"
            userinfo = {} # user decisions
            # your decision here
            if float(res["typeid"]) == 1:   # 1=temperature , 2=Pressure, 3=heart puls عشان ما نتعامل بالاسماء
                if res["mvalue"] != "":
                    if float(res["mvalue"]) > 37:
                        userinfo["username"] = res["username"]
                        userinfo["mob"] = res["mob"]
                        userinfo["desc"] = "too high"
                        webDataInfos.append(userinfo)
            if float(res["typeid"]) == 2:    
                if res["mvalue"] != "":
                    if float(res["mvalue"]) > 120:
                        userinfo["username"] = res["username"]
                        userinfo["mob"] = res["mob"]
                        userinfo["desc"] = "too high"
                        webDataInfos.append(userinfo)
            if float(res["typeid"]) == 3:   
                if res["mvalue"] != "":
                    if float(res["mvalue"]) > 100:
                        userinfo["username"] = res["username"]
                        userinfo["mob"] = res["mob"]
                        userinfo["desc"] = "too high"
                        webDataInfos.append(userinfo)
        
        print(webDataInfos)
        HighTemps = ""
        for webDa in webDataInfos:
            HighTemps += str(webDa["username"]) + " " + str(webDa["mob"]) + " " + str(webDa["desc"])+ ", "
        self.label_des.setText("Users with High temperature: \n " + HighTemps)
        self.label_des.setStyleSheet("QLabel { background-color : red; color : black; }");
        self.label_database.setText(textvalue)

    # Convert an opencv image to QPixmap
    def drawPrevImage(self, cvImg, imgTitle):
        # Convert image to QImage
        scene = QtWidgets.QGraphicsScene()
        pixmap = QImage(cvImg.data, cvImg.shape[1], cvImg.shape[0], QImage.Format_RGB888).rgbSwapped()
        pix = QPixmap(pixmap)
        scene.addPixmap(pix)
        self.graphicsView_prev.setScene(scene)   # set the current viewed image

    # Convert an opencv image to QPixmap
    def drawResImage(self, cvImg, imgTitle):
        # Convert image to QImage
        scene = QtWidgets.QGraphicsScene()
        pixmap = QImage(cvImg.data, cvImg.shape[1], cvImg.shape[0], QImage.Format_RGB888).rgbSwapped()
        pix = QPixmap(pixmap)
        scene.addPixmap(pix)
        self.graphicsView_res.setScene(scene)   # set the current viewed image
    
    def Browser(self):
        method = self.comboBox.currentText()
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        ImageFile = QtWidgets.QFileDialog.getOpenFileName(self, "Select Image To Process", "","All Files (*);;Image Files(*.jpg *.gif)", options=options)
        if ImageFile:
            self.label_filename.setText(ImageFile[0])
            if(method != "Video"):
                org_img = cv2.imread(ImageFile[0]) 
                img_colored = cv2.resize(org_img ,(400,400))    
                self.drawPrevImage(img_colored , "Original")
            
    def trainSVMdetect(self, frame):
        orgframe = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)    
        bounding_box_cordinates, weights =  HOGCV.detectMultiScale(frame, 
                                        winStride=(1, 1),
                                        padding=(8, 8),
                                        scale=1.05) 
        print(bounding_box_cordinates)
        blank_image = np.zeros((224,224))
        person = 1
        for x,y,w,h in bounding_box_cordinates:
            cv2.rectangle(orgframe, (x,y), (x+5,y+5), (0,255,0), 8)
            cv2.putText(blank_image, f'person {person}', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            person += 1
        
        return blank_image # number of persons
    
    def classifier(self):
        human_dir = "human"
        nohuman_dir = "nohuman"
        
        df = pd.DataFrame() #hold the image arrays
        y_values = []  # hold the image labels (classes)
        
        
        count=0
        for img_name in os.listdir(human_dir):
            img = cv2.imread(human_dir+"/"+img_name)
            img = cv2.resize(img ,(224,224))
            #img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)        
            #img_one_column_arr = img_gray.reshape((img.shape[0]*img.shape[1]), 1)
            img = self.trainSVMdetect(img)
            img_one_column_arr = img.reshape((img.shape[0]*img.shape[1]), 1)
            df[str(count)] = img_one_column_arr[:,0]
            y_values.append(1) # 0 = human
            count += 1
        
        for img_name in os.listdir(nohuman_dir):
            img = cv2.imread(nohuman_dir+"/"+img_name)
            img = cv2.resize(img ,(224,224))
            #img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #img_one_column_arr = img_gray.reshape((img.shape[0]*img.shape[1]),1) 
            #df[str(count)] = img_one_column_arr[:,0]
            img = self.trainSVMdetect(img)
            img_one_column_arr = img.reshape((img.shape[0]*img.shape[1]), 1)
            df[str(count)] = img_one_column_arr[:,0]
            y_values.append(0) # 1 == nohuman
            count += 1
        
        x_data = df.T
        
        #Split the dataset into training and test sets 
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_values, test_size=0.3, stratify=y_values, random_state=42)
        
        scaler = StandardScaler()
        scaler.fit(x_train)
        #x = tf.Variable([[1.], [2.]])
        #y = tf.Variable([[1.], [2.]])
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        #tf.placeholder(tf.float32, [None, dim])
        #tf.Variable(tf.zeros([dim, n_classes]))
        #tf.Variable(tf.zeros([n_classes]))


        
        
        f = open('SVMresult.txt', 'w')
        '''
        SVM
        '''
        print("Classifier: SVM" )
        f.write("Classifier: SVM\n")
        
        svclassifier = SVC(kernel='linear')
        svclassifier.fit(x_train, y_train)
        
        y_pred = svclassifier.predict(x_test)
        print("Confusion Matrix:")
        f.write("Confusion Matrix:\n")
        print(confusion_matrix(y_test,y_pred))
        f.write(str(confusion_matrix(y_test,y_pred)) + "\n")
        print("Report:")
        f.write("Report\n")
        print(classification_report(y_test,y_pred))
        f.write(str(classification_report(y_test,y_pred)) + "\n")
        f.close()
        contents = ""
        with open('SVMresult.txt') as fr:
            contents += fr.read()
        self.label_classifierRep.setText(contents)

        summation = 0  #variable to store the summation of differences
        n = len(y_test) #finding total number of items in list
        for i in range (0,n):  #looping through each element of the list
            difference = y_test[i] - y_pred[i]  #finding the difference between observed and predicted value
            squared_difference  = difference**2  #taking square of the differene 
            summation = summation + squared_difference  #taking a sum of all the differences
        MSE = 0 
        if n > 0:
            MSE = summation/n  #dividing summation by total values to obtain average
        meanerror = "The Mean Square Error is: " + str(MSE)
        self.label_classifierRep.setText(meanerror)
        
        plt.plot(y_test, y_pred )
        plt.xlabel('Input Values')
        plt.ylabel('Predict Values')
        plt.show()
        print (meanerror )

        '''
        theta_values = np.linspace(0, 6, 244)
        mse = (np.square(x_test, y_pred)).mean(axis=-1) # same as line 175
        plt.figure()
        plt.plot(theta_values, mse)
        plt.xlabel('Theta Values')
        plt.ylabel('Squared Error')
        plt.show()
        '''
        
       

        #loss = mean(square(y_test - y_pred), axis=-1)
       # plot(mse)
    
        
    # Alaram & Desscion Taken ----------------------------
    def fireAlarm(self, personcount):
        z="http://cmsat.atwebpages.com/measures.php"
        result = "Normal"
        colr = "No DANGEROUS SITUATION"
        self.label_des.setStyleSheet("QLabel { background-color : green; color : black; }");
        if(personcount > 5) :
            result = "Medium stat"
            colr = "The CubeSat Made an  Predefined Action"
            health=" there is an emergency Health situation"
            self.label_des.setStyleSheet("QLabel { background-color : yellow; color : black; }");
        if(personcount > 7):
            result = "Alarm"
            colr = "Adman Must Take The Action!!"
            self.label_des.setStyleSheet("QLabel { background-color : red; color : black; }");
        self.label_des.setText(colr)
        
    #Analysis pictures and video as frames----------------------------------------------------------p
    def detect(self, frame):
        orgframe = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)    
        bounding_box_cordinates, weights =  HOGCV.detectMultiScale(frame, 
                                        winStride=(4, 4),
                                        padding=(16, 16),
                                        scale=1.05) 
        person = 1
        for x,y,w,h in bounding_box_cordinates:
            cv2.rectangle(orgframe, (x,y), (x+5,y+5), (0,255,0), 1)
            cv2.putText(orgframe, f'person {person}', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            person += 1
        
        self.label_output.setText("Persons in image = " + str(len(bounding_box_cordinates)))
        cv2.putText(orgframe, 'Status : Detecting ', (40,40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
        cv2.putText(orgframe, f'Total Persons : {person-1}', (40,70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)

        self.fireAlarm(person)
        return orgframe  
    
    # If we want to detect from picture ------------We use: python FileName.py -i PicName.jpeg or .png
    def detectByPathImage(self, path):
       # head_cascade = cv2.CascadeClassifier('Head6.xml')
        image = cv2.imread(path)
        image = imutils.resize(image, width = min(1200, image.shape[1]))
        result_image = self.detect(image)
        self.drawResImage(result_image, "Result")   # set the current viewed image        
    
    # another way using DLIB
    def detectByImage(self, path):
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        self.label_output.setText("Faces in image = " + str(len(faces)))
        self.fireAlarm(len(faces))
        self.drawResImage(img, "face results")   # set the current viewed image        
    
    
        # If we want to detect picture from video ------------We use: python FileName.py -v VedioFileName.mp4------Then press c
    def detectByCamera(self):
        cap = cv2.VideoCapture(0)
        #train_alg = cv2.CascadeClassifier('Head6.xml')

    
        # Check if the webcam is opened correctly
        if not cap.isOpened():
            raise IOError("Cannot open webcam")
    
        while True:
            ret, frame = cap.read()
            frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            writer = cv2.VideoWriter("output.avi",cv2.VideoWriter_fourcc(*'MJPG'), 10, (400,400))

            c = cv2.waitKey(1)
            if ret:
                frame = imutils.resize(frame , width=min(2200,frame.shape[1]))
                frame = self.detect(frame)
                frame = imutils.resize(frame , width=min(600,frame.shape[1]))
            if c == ord(' '):
                continue
            if c == ord('q'):
                frame = self.detect(frame)
                self.drawResImage(frame, "frame")
                cap.release()
                cv2.destroyAllWindows()
                 

                break
                
            cv2.imshow('Input', frame)

            if c == 27:
                break
    
    #If we want to detect from vedio ------------We use: python FileName.py -v VedioFileName.mp4
    def detectByPathVideo(self, path):
        print(path)
        video = cv2.VideoCapture(path)
        check, frame = video.read()
        if check == False:
            print('Video Not Found. Please Enter a Valid Path (Full path of Video Should be Provided).')
            return
        print('Detecting people...')
        while video.isOpened():
            #check is True if reading was successful 
            check, frame =  video.read()
            if check:
                frame = imutils.resize(frame , width=min(2200,frame.shape[1]))
                frame = self.detect(frame)
                frame = imutils.resize(frame , width=min(600,frame.shape[1]))
                cv2.imshow('Result', frame)
                
                
                key = cv2.waitKey(1)
                if key== ord('c'):
                    cv2.imshow("file", frame)
                    cv2.imwrite("mypic.jpg", frame)
    
                if key== ord('q'):
                    break
            else:
                break
        video.release()
        cv2.destroyAllWindows()    
          
    def startDetect(self):
        method = str(self.comboBox.currentText())
        filepath = self.label_filename.text()
        filecheck = str(self.label_filename.text())
        self.label_output.setText("Starting ...")
        if(method != "Camera"):
            if(filecheck == ""):
                self.label_output.setText("Please select a method")
            else:   
                self.label_output.setText("Start detectiong ... Please wait")
                if(method == "Image body"):
                    self.detectByPathImage(filepath)
                if(method == "Image face"):
                    self.detectByImage(filepath)
                if(method == "Video"):
                    #os.system('python hog2.py -v crowd.mp4')
                    #exec(open("vidcam.py -v " + filepath).read())
                    #os.system('python vidcam.py -v "' + filepath + '"')
                    self.detectByPathVideo(filepath)
        else:
            if(method == "Camera"):
                self.detectByCamera()
  
    def Close(self):
        self.destroy()
    
    
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())