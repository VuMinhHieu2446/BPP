import serial
arduino = serial.Serial(port='COM5', baudrate=9600, timeout=.1)
import ssl                          # Establish secure connection
import sys
import paho.mqtt.client as mqtt     # Connect with the MQTT Library
import time                         # Time Library
import threading
from tkinter import ttk, font       # Import Tkinter Lybrary
import tkinter
import getpass
import random
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg#, NavigationToolbar2TkAgg
from matplotlib.pyplot import figure
import  matplotlib.pyplot as plt

root = tkinter.Tk()
root.geometry('640x480')#size window
style = ttk.Style(root)
style.configure('lefttab.TNotebook', tabposition='wn')

tabControl = tkinter.ttk.Notebook(root, style='lefttab.TNotebook')
root.title("Cảm biến và xử lý tín hiệu")

tab1 = tkinter.ttk.Frame(tabControl)
tab2 = tkinter.ttk.Frame(tabControl)
tab3 = tkinter.ttk.Frame(tabControl)

#tabControl.add(tab1, text ='login')
tabControl.add(tab2, text ='plot ')
#tabControl.add(tab3, text ='blink')
tabControl.pack(expand = 1, fill ="both")



########################## tab 2 ##########################
#plotter
thoiGianDemXung = 0
soXung = []
soXungKalman = []

continuePlotting = False

def change_state(): 
    global continuePlotting
    if continuePlotting == True: 
        continuePlotting = False 
    else:
        #x = str(int(text3.get())*98*60/1000).split('.')[0]
        x=str(text3.get())
        y=str(text0.get())
        print(x)
        arduino.write(bytes(x, 'utf-8'))
        continuePlotting = True
    
fig = plt.figure(figsize=(1, 2), dpi=100)
#fig = figure(figsize=(1, 1), dpi=100) 
     
ax = fig.add_subplot() 
ax.grid()
ax.axis([0, 35, 0, 30])
 
graph = FigureCanvasTkAgg(fig, master=tab2) 
graph.get_tk_widget().pack(side = "top", fill='both',expand=False) 

def xuLydata():
    global soXung, soXungKalman
    value = arduino.readline()
    if value!=b'':
        print(str(value))
        value = str(value)[2:-5]
        value = value.split('+')
        #print(value[0])
        #print(value[1])
        #if len(value) == 2:
        soXung.append(int(value[0]))
        soXungKalman.append(int(value[1]))

def max_(soXung):
    if len(soXung)==0:
        temp = 30
    else:
        temp=max(soXung)
    return temp
    
def plotter():
    global thoiGianDemXung, soXung, soXungKalman, continuePlotting
    thoiGianDemXung = int(text0.get())
    while continuePlotting:
        xuLydata()
        text1.set(str(sum(np.array(soXungKalman)/98/thoiGianDemXung*1e3/len(soXungKalman)))[:5])
        text2.set(str(sum(np.array(soXungKalman)/98*1e3/60))[:5])
        ax.cla() 
        ax.grid()
        n=len(soXung)
        ax.axis([0, int(n/35)*35+35, 0, int(max_(soXung)/30)*30+30])
        ax.plot(range(n), soXung, label='k kalman')#marker='o', color='orange')
        ax.plot(range(n), soXungKalman, label='có dùng kalman')
        ax.legend(fontsize = 'x-small')
        graph.draw() 
        #time.sleep(1) 

def gui_handler():
    global thoiGianDemXung
    thoiGianDemXung = int(text0.get())
    print('thoi gian dem xung: ', thoiGianDemXung)
    change_state()
    t2=threading.Thread(target=plotter)
    t2.start()

def resetThreading():
    t1=threading.Thread(target=reset)
    t1.start()

def reset():
    global soXung, soXungKalman, continuePlotting
    #arduino.write(bytes('-1', 'utf-8'))
    continuePlotting = False
    soXung =[]
    soXungKalman =[]
    print(text2.get())
    text0.set('100')
    text1.set('')
    text2.set('')
    text3.set('100')
    
    ax.cla() 
    ax.grid()
    ax.axis([0, 35, 0, 30])
    ax.plot(range(len(soXung)), soXung, label='k kalman')#marker='o', color='orange')
    ax.plot(range(len(soXungKalman)), soXungKalman, label='có dùng kalman')
    ax.legend(fontsize = 'x-small')
    graph.draw() 
    
def save():
    print(text3.get())
    pass

text0 = tkinter.StringVar()
L0 = tkinter.Label(root, text="Thời gian đếm xung: ")
L0.place(x=50, y=220)
E0 = tkinter.Entry(root, bd =2, textvariable=text0, justify='center')
text0.set('200')
E0.place(x=200, y=220, width=70)
L01 = tkinter.Label(root, text="(ms)")
L01.place(x=280, y=220)

text1 = tkinter.StringVar()
L1 = tkinter.Label(root, text="Lưu lượng: ")
L1.place(x=70, y=270)# width=50, height=50)
E1 = tkinter.Entry(root, bd =2, textvariable=text1, justify='center')
E1.place(x=200, y=270, width=70)# width=50, height=50)
L11 = tkinter.Label(root, text="(lít/phút)")
L11.place(x=280, y=270)# width=50, height=50)

text2 = tkinter.StringVar()
L2 = tkinter.Label(root, text="Thể tích: ")
L2.place(x=70, y=320)# width=50, height=50)
E2 = tkinter.Entry(root, bd =2, textvariable=text2, justify='center')
E2.place(x=200, y=320, width=70)
L21 = tkinter.Label(root, text="(ml)")
L21.place(x=280, y=320)

text3 = tkinter.StringVar()
L3 = tkinter.Label(root, text="Thể tích yêu cầu: ")
L3.place(x=50, y=370)# width=50, height=50)
E3 = tkinter.Entry(root, bd =2, textvariable=text3, justify='center')
E3.place(x=200, y=370, width=70)
text3.set('200')
L31 = tkinter.Label(root, text="(ml)")
L31.place(x=280, y=370)

buttonStart = tkinter.Button(tab2, text="Start/Stop", command=gui_handler)#, bg="red", fg="white") 
buttonStart.place(x=50, y=420)
buttonReset = tkinter.Button(tab2, text="Reset", command=resetThreading)#, bg="red", fg="white") 
buttonReset.place(x=150, y=420)
buttonSave = tkinter.Button(tab2, text="Save", command=save, bg="blue", fg="white")
buttonSave.place(x=500, y=430)

root.mainloop() 
