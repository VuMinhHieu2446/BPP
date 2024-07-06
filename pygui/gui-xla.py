import paho.mqtt.client as mqtt  
import numpy as np
from PIL import ImageGrab
import time
import tkinter
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import threading
def fd_histogram(image,bins=16, mask=None):
    # convert the image to HSV color-space
    img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([img_hsv], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])

    #hist = cv2.calcHist([img_hsv], [0, 1, 2], None, [256], [0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()
    #return(hist)

def test_model(imageOrigin, model):
    #imageOrigin = cv2.imread(path)
    image = imageOrigin.copy()
    contours = find_contours_ov2640(image)

    # Loop over each contour
    for (i, c) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(c)
        if w*h<3000:
            continue
            
        coin = imageOrigin[y:y + h, x+const:x+const + w]
        #cv2.imshow(str(i)+' jpg', coin)
        #cv2.waitKey(0)
        resized = cv2.resize(coin, [32, 32], interpolation = cv2.INTER_AREA)
        
        #print(resized.shape)
        #resized = np.reshape(resized, (1, resized.shape[0]*resized.shape[1]*3))
        #resized=resized[0]
        #resized = np.hstack((resized, np.array((1))))
        #print(resized.shape)
        predicted = model.predict([fd_histogram(resized)])
        print(predicted)

import pickle
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

const = 0
def find_contours_ov2640(image):
    global const
    image = image[0:640, const:480]
    image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])#cân bằng sáng
    image[:, :, 1] = cv2.equalizeHist(image[:, :, 1])
    image[:, :, 2] = cv2.equalizeHist(image[:, :, 2])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)
    ret, thresh1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    edged = cv2.Canny(blurred, 30, 100)
    #cv2.imshow("Image", edged)
    #rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    #dilation = cv2.dilate(edged, rect_kernel, iterations = 1)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #coins = image.copy()
    #cv2.drawContours(coins, contours, -1, (0, 255, 0), 1)
    #cv2.imshow("Coins", coins)
    #cv2.waitKey(0)
    return contours

def detect_fruit(imageOrigin):
    image = imageOrigin.copy()
    contours = find_contours_ov2640(image)

    # Loop over each contour
    for (i, c) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(c)
        if w*h<3000:
            continue
            
        coin = imageOrigin[y:y + h, x+const:x+const + w]

        resized = cv2.resize(coin, [32, 32], interpolation = cv2.INTER_AREA)
        predicted = model.predict([fd_histogram(resized)])
        print(predicted)
        
        start_point = [x+const, y]
        end_point = [x + w +const, y + h]
        color = [255, 255, 255]
        thickness = 1
        imageOrigin = cv2.rectangle(imageOrigin, start_point, end_point, color, thickness)
        if '1' in str(predicted):
            text = 'xanh'
        elif '0' in str(predicted):
            text = 'chin'
        else:
            text = 'no detect'
        cv2.putText(imageOrigin, text, (x+60,y-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, color, 
                   thickness, cv2.LINE_AA)
    #cv2.imshow('image', imageOrigin)
    return imageOrigin
    
root = tkinter.Tk()
root.geometry('1000x650')#size window

lmain = tkinter.Label(root)
lmain.place(x=20, y=20)#, width=50, height=50)
#lmain.pack()

streamVideo=True
def threadingStartStreamVideo():
    global streamVideo
    streamVideo=True
    client.publish("turnCamera", '{"turnCamera":1}', qos=2)
    t1=threading.Thread(target=startStreamVideo)
    t1.start()

nameSelectedModel=''
stt=0
sttAnhChup = 0
stt_detect = 0
def startStreamVideo():
    global streamVideo, buff, imgready, stt, sttAnhChup
    if(imgready == True):
        #print('2')
        nparr = np.frombuffer(buff, np.uint8)
        cv2image = cv2.imdecode(nparr,1)
        if stt==1:
            cv2.imwrite('.\\data-raw\\image_'+str(sttAnhChup)+'.png',cv2image)
            sttAnhChup = sttAnhChup+1
            stt=0
            
        
        if nameSelectedModel == 'detect fruit':
            cv2image=detect_fruit(cv2image)
        cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        if streamVideo:
            lmain.after(20, startStreamVideo)


def stopStreamVideo():
    global streamVideo
    streamVideo = False
    client.publish("turnCamera", '{"turnCamera":0}', qos=2)

def threadingChup():
    client.publish("SMILE", '1', qos=2)
    t1=threading.Thread(target=chup)
    t1.start()

import time
def chup():
    global streamVideo, buff, imgready, stt, sttAnhChup
    if imgready == True:
        nparr = np.frombuffer(buff, np.uint8)
        img = cv2.imdecode(nparr,1)
        #cv2.imshow('timg', img)
        #cv2.waitKey(0)
        cv2.imwrite('.\\data-raw\\image_'+str(sttAnhChup)+'.png',img)
        del nparr
        sttAnhChup = sttAnhChup+1
    imgready = False

    
#Create Button
buttonStart = tkinter.Button(root,text="Start",command = threadingStartStreamVideo)
buttonStart.place(x=700, y=150)
buttonStop = tkinter.Button(root,text="Stop",command = stopStreamVideo)
buttonStop.place(x=800, y=150)

buttonChup = tkinter.Button(root,text="Chụp ảnh",command = threadingChup)
buttonChup.place(x=800, y=300)

# Combobox creation
L1 = tkinter.Label(root, text="FRAME SIZE")
L1.place(x=700, y=200)# width=50, height=50)

frameSize = tkinter.StringVar()
frameSizeCombobox = ttk.Combobox(root, width = 27, textvariable = frameSize)
frameSizeCombobox.place(x=800, y=200, width=120)#, height=50)
#FRAMESIZE_VGA
#QVGA|CIF|VGA|SVGA|XGA|SXGA|UXGA
frameSizeCombobox['values'] = ('FRAMESIZE_QVGA', 'FRAMESIZE_VGA',
                               'FRAMESIZE_SVGA','FRAMESIZE_XGA',
                               'FRAMESIZE_SXGA','FRAMESIZE_UXGA')
frameSizeCombobox.current(1)


L2 = tkinter.Label(root, text="Time Delay")
L2.place(x=700, y=250)# width=50, height=50)
timeFrame = tkinter.Entry(root, bd =5)
timeFrame.insert(tkinter.END, '100')
timeFrame.place(x=800, y=250)# width=50, height=50)

def threadingPublishConfig():
    t1=threading.Thread(target=publishConfig)
    t1.start()
def publishConfig():
    print('frameSize: ', frameSize.get())
    print('timeFrame: ', timeFrame.get())
    client.publish("JSONConfig", '{"framesize":"'+str(frameSize.get())+'", "delayTime":'+str(timeFrame.get())+'}', qos=2)
    #time.sleep(5000)

buttonPublish = tkinter.Button(root,text="Publish",command = threadingPublishConfig)
buttonPublish.place(x=700, y=300)

L1 = tkinter.Label(root, text="Model")
L1.place(x=700, y=350)# width=50, height=50)
modelName = tkinter.StringVar()
selectModel = ttk.Combobox(root, width = 27, textvariable = modelName)
selectModel.place(x=800, y=350, width=120)
selectModel['values'] = ('No', 'detect fruit')
selectModel.current(0)


'''def SelectModel():
    global modelName, nameSelectedModel
    nameSelectedModel = modelName.get()

buttonSelect = tkinter.Button(root,text="Select Model",command = SelectModel)
buttonSelect.place(x=700, y=400)'''

def model_changed(event):
    global modelName, nameSelectedModel
    nameSelectedModel = modelName.get()
    print(selectModel.get())

selectModel.bind('<<ComboboxSelected>>', model_changed)


buff=[]
imgready = False
sensorData1 = []
sensorData2 = []

def on_connect(client, userdata, flags, rc):
    print("Connected With Result Code "+str(rc))
    
def on_publish(client, obj, mid):
    print("mid: "+str(mid))

def on_message(client, userdata, message):
    global buff
    global imgready
    print("message received ")
    imgready = True
    buff = message.payload

broker_address="broker.mqttdashboard.com" #Replace with your broker adress
broker_port = 1883
client = mqtt.Client()
client.on_message = on_message
client.on_connect = on_connect
client.on_publish = on_publish
client.username_pw_set("jazz23", "12345") #Replace with your User/Pass
client.connect(broker_address, broker_port)

#client.publish("JSONConfig", '{"framesize":"FRAMESIZE_VGA"}', qos=2)

client.subscribe("PICTURE", 0) #Replace with your Topic
threading.Thread(target=client.loop_forever).start()

root.mainloop()
