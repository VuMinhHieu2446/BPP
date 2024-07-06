import ssl                          # Establish secure connection
import sys
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
from PIL import Image, ImageTk
import cv2
from object_3d import *
from camera import *
from projection import *

class SoftwareRender:
    def __init__(self):
        pg.init()
        self.RES = self.WIDTH, self.HEIGHT = 800, 450
        self.H_WIDTH, self.H_HEIGHT = self.WIDTH // 2, self.HEIGHT // 2
        self.FPS = 60
        # self.screen = pg.display.set_mode(self.RES)
        # self.clock = pg.time.Clock()
        self.create_objects()

    def create_objects(self):
        self.camera = Camera(self, [0.5, 1, -4])
        self.projection = Projection(self)
        self.object = Object3D(self)
        #self.object.translate([0.2, 0.4, 0.2])
        self.object.rotate_y(math.pi/6)

    def draw(self):
        #self.screen.fill(pg.Color('darkslategray'))
        self.object.draw()

    def run(self):
        while True:
            self.draw()
            # self.camera.control()
            # [exit() for i in pg.event.get() if i.type == pg.QUIT]
            # pg.display.set_caption(str(self.clock.get_fps()))
            # pg.display.flip()
            # self.clock.tick(self.FPS)
            img = Image.fromarray(app.object.frame)
            imgtk = ImageTk.PhotoImage(image=img)
            lmain.imgtk = imgtk
            lmain.configure(image=imgtk)
            if streamVideo:
                lmain.after(20, startStreamVideo)

root = tkinter.Tk()
#getting screen width and height of display
width= root.winfo_screenwidth()
height= root.winfo_screenheight()
#setting tkinter window size
root.geometry("%dx%d" % (width, height))
style = ttk.Style(root)
style.configure('lefttab.TNotebook', tabposition='wn')

tabControl = tkinter.ttk.Notebook(root, style='lefttab.TNotebook')
root.title("Cảm biến và xử lý tín hiệu")

tab1 = tkinter.ttk.Frame(tabControl)
tab2 = tkinter.ttk.Frame(tabControl)
tab3 = tkinter.ttk.Frame(tabControl)

#tabControl.add(tab1, text ='login')
tabControl.add(tab1, text ='image')
tabControl.add(tab2, text ='plot ')
#tabControl.add(tab3, text ='blink')
tabControl.pack(expand = 1, fill ="both")

########################## tab 2 ##########################
lmain = tkinter.Label(root)
lmain.place(x=50, y=20)#, width=50, height=50)
streamVideo = True
app = SoftwareRender()
def startStreamVideo():
    global streamVideo, app
    cv2image = cv2.imread('D:/New folder (5)/20212/do-an-20221/New folder (4)/pygui/Python.png')
    scale_percent = 10 # percent of original size
    width = int(cv2image.shape[1] * scale_percent / 100)
    height = int(cv2image.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # resize image
    cv2image = cv2.resize(cv2image, dim, interpolation = cv2.INTER_AREA)
    #cv2.imshow('tung', cv2image)

    
    app.draw()

    img = Image.fromarray(app.object.frame)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    if streamVideo:
        lmain.after(20, startStreamVideo)

def threadingStartStreamVideo():
    global streamVideo
    streamVideo=True
    t1=threading.Thread(target=startStreamVideo)
    t1.start()

buttonStart = tkinter.Button(root,text="Start",command = threadingStartStreamVideo)
buttonStart.place(x=300, y=150)



root.mainloop()