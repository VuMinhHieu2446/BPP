import numpy as np
from mayavi.mlab import *
import threading
import DobotDllType as dType
import mayavi.mlab as mlab
import sys
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5 import uic
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import random

import time
import cv2
import cv2 as cv
from pylibdmtx import pylibdmtx
import threading
from model import *
from tools import *
import givenData
import gym
import torch.nn as nn
from tools import init
import tools
from numpy import sqrt
from attention_model import AttentionModel
import numpy as np
from pct_envs.PctDiscrete0.binCreator import RandomBoxCreator, LoadBoxCreator, BoxCreator

# mlab.clf()
def draw_box(box):
    x, y, z = box.lx, box.ly, box.lz
    dx,dy,dz = box.x, box.y, box.z
    color_box = tuple(np.random.rand(1, 3)[0])
    #x = 10; y = 10; z = 10; dx = 10; dy = 10; dz = 10
    Y, Z= np.mgrid[y : y+dy : 10j, z : z+dz : 10j]
    X = np.ones([10, 10])*x
    mlab.mesh(X, Y, Z, color=color_box)
    mlab.mesh(X+dx, Y, Z, color=color_box)

    Y, X= np.mgrid[y : y+dy : 10j, x : x+dx : 10j]
    Z = np.ones([10, 10])*z
    mlab.mesh(X, Y, Z, color=color_box)
    mlab.mesh(X, Y, Z+dz, color=color_box)

    Z, X= np.mgrid[z : z+dz : 10j, x : x+dx : 10j]
    Y = np.ones([10, 10])*y
    mlab.mesh(X, Y, Z, color=color_box)
    mlab.mesh(X, Y+dy, Z, color=color_box)


# draw_box(0, 0, 0, 1, 1, 1, color_box = tuple(np.random.rand(1, 3)[0]))
# draw_box(0, 0, 1, 1, 1, 1, color_box = (0, 0, 1))
# draw_box(1, 1, 1, 2, 2, 2, color_box = (1, 0, 0))
# mlab.show()

# f = mlab.gcf()
# f.scene._lift()
# arr = mlab.screenshot()
# import pylab as pl
# pl.imshow(arr)
# pl.axis('off')
# pl.show()

class DRL_GAT(nn.Module):
    def __init__(self, args_embedding_size,
                 args_hidden_size,
                 args_gat_layer_num,
                 args_internal_node_holder,
                 args_internal_node_length,
                 args_leaf_node_holder):
        super(DRL_GAT, self).__init__()

        self.actor = AttentionModel(args_embedding_size,
                                    args_hidden_size,
                                    n_encode_layers = args_gat_layer_num,
                                    n_heads = 1,
                                    internal_node_holder = args_internal_node_holder,
                                    internal_node_length = args_internal_node_length,
                                    leaf_node_holder = args_leaf_node_holder,
                                    )
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), sqrt(2))
        self.critic = init_(nn.Linear(args_embedding_size, 1))

    def forward(self, items, deterministic = False, normFactor = 1, evaluate = False):
        o, p, dist_entropy, hidden, _= self.actor(items, deterministic, normFactor = normFactor, evaluate = evaluate)
        values = self.critic(hidden)
        return o, p, dist_entropy,values

    def evaluate_actions(self, items, actions, normFactor = 1):
        _, p, dist_entropy, hidden, dist = self.actor(items, evaluate_action = True, normFactor = normFactor)
        action_log_probs = dist.log_probs(actions)
        values =  self.critic(hidden)
        return values, action_log_probs, dist_entropy.mean()

registration_envs()

class initi_envs:
    def __init__(self):
        pass

class run_rl_model():
    def __init__(self, app, stt_model):
        self.app = app
        self.stt_model = stt_model
        self.id = 'PctDiscrete-v0'
        if self.stt_model == 1:
            self.setting = int(self.app.comboBox_setting_1.currentText()[-1])
            self.lnes = self.app.comboBox_lnes_1.currentText()[:-1]
        elif self.stt_model == 2:
            self.setting = int(self.app.comboBox_setting_2.currentText()[-1])
            self.lnes = self.app.comboBox_lnes_2.currentText()[:-1]
        else:
            self.setting = int(self.app.comboBox_setting_actual.currentText()[-1])
            self.lnes = self.app.comboBox_lnes_actual.currentText()[:-1]
        # self.setting = 2
        if self.setting == 1:
            self.internal_node_length = 6
            self.model_path = "PCB-model/setting1_discrete.pt"
        elif self.setting == 2:
            self.internal_node_length = 6
            self.model_path = "PCB-model/setting2_discrete.pt"
        elif self.setting == 3:
            self.internal_node_length = 7
        self.container_size = self.app.size_con#givenData.container_size
        self.item_size_set  = givenData.item_size_set
        self.dataset_path = "PCB-dataset/setting123_discrete.pt"
        self.load_dataset = True
        self.internal_node_holder = 80 #Maximum number of internal nodes
        self.leaf_node_holder = 50 #Maximum number of leaf nodes
        #self.lnes = 'EMS' #Leaf Node Expansion Schemes: EMS (recommend), EV, EP, CP, FC
        # self.lnes = self.app.comboBox_lnes_1.currentText()[:-1] if self.stt_model==1 else \
        #     self.app.comboBox_lnes_2.currentText()[:-1]
        self.shuffle = True #Randomly shuffle the leaf nodes
        self.num_processes = 1 #The number of parallel processes used for training
        self.device = 'cpu'
        device = self.device
        self.envs = gym.make(self.id,
                        setting=self.setting,
                        container_size=self.container_size,
                        item_set=self.item_size_set,
                        data_name=self.dataset_path,
                        load_test_data=self.load_dataset,
                        internal_node_holder=self.internal_node_holder,
                        leaf_node_holder=self.leaf_node_holder,
                        LNES=self.lnes,
                        shuffle=self.shuffle)

        self.embedding_size = 64 #Dimension of input embedding
        self.hidden_size = 128
        self.gat_layer_num = 1
        self.PCT_policy =  DRL_GAT(self.embedding_size,
                        self.hidden_size,
                        self.gat_layer_num,
                        self.internal_node_holder,
                        self.internal_node_length,
                        self.leaf_node_holder)

        #self.model_path = "D:/New folder (5)/20212/do-an-20221/all_model/PCT-TUNG-2023.02.04-07-07-37_2023.02.04-07-08-46.pt"
        #'D:/New folder (5)/20212/do-an-20221/New folder/PCB-model/setting2_discrete.pt'
        #"D:/New folder (5)/20212/do-an-20221/New folder (4)/PCB-model/PCT-model3_tung-2022.11.22-20-40-06_2022.11.22-20-41-14.pt"
        self.PCT_policy = load_policy(self.model_path, self.PCT_policy)
        print('Pre-train model loaded!')
        self.envs.type_gen_box = "Manual"
        self.envs.type_gen_box = "Auto"
        self.PCT_policy.eval()
        self.envs.next_box_cf = (5, 5, 5)#first box
        self.done = self.envs.reset()
        #self.obs = torch.FloatTensor(self.obs).to(self.device).unsqueeze(dim=0)
        #self.all_nodes, self.leaf_nodes = tools.get_leaf_nodes_with_factor(self.obs, self.num_processes, self.internal_node_holder, self.leaf_node_holder)

        self.batchX = torch.arange(self.num_processes)
        self.step_counter = 0
        self.episode_ratio = []
        self.episode_length = []
        self.all_episodes = []

        self.evaluation_episodes = 1 #Number of episodes evaluated
        self.eval_freq = self.evaluation_episodes
        self.normFactor = 1.0 / np.max(self.container_size)
        self.factor = self.normFactor
        self.frames = []

        self.stt_box_random = 0

    def run_step(self, *arg):
        if self.envs.type_gen_box == "Manual":
            self.envs.next_box_cf = (float(self.app.lineEdit_X.text())/10,
                                  float(self.app.lineEdit_Y.text())/10,
                                  float(self.app.lineEdit_Z.text())/10)
            print(self.stt_model, self.envs.next_box_cf)
        else:
            self.envs.next_box_cf = self.app.randombox
        self.obs = self.envs.cur_observation()
        self.obs = torch.FloatTensor(self.obs).to(self.device).unsqueeze(dim=0)
        self.all_nodes, self.leaf_nodes = tools.get_leaf_nodes_with_factor(self.obs, self.num_processes,
                                                                            self.internal_node_holder, self.leaf_node_holder)
        with torch.no_grad():
            selectedlogProb, selectedIdx, policy_dist_entropy, value = self.PCT_policy(self.all_nodes, True, normFactor = self.factor)
        selected_leaf_node = self.leaf_nodes[self.batchX, selectedIdx.squeeze()]
        self.items = self.envs.packed
            
        done, reward, self.done, infos = self.envs.step(selected_leaf_node.cpu().numpy()[0][0:6], (5, 5, 5))#, my_next_box)
        self.box = self.envs.space.boxes#[:-1]
        box = self.box[-1]
        x, y, z = box.lx, box.ly, box.lz
        dx,dy,dz = box.x, box.y, box.z
        print(self.stt_model, x, y, z, dx, dy, dz)
        #self.app.pick_and_place_item([x/10, y/10, z/10]) #control robot

        if self.done:
            print('Episode {} ends.'.format(self.step_counter))
            if 'ratio' in infos.keys():
                self.episode_ratio.append(infos['ratio'])
            if 'counter' in infos.keys():
                self.episode_length.append(infos['counter'])

            print('Mean ratio: {}, length: {}'.format(np.mean(self.episode_ratio), np.mean(self.episode_length)))
            print('Episode ratio: {}, length: {}'.format(infos['ratio'], infos['counter']))
            self.all_episodes.append(self.items)
            self.step_counter += 1
            # self.obs = self.envs.reset()               

    def run_complete(self):
        while not self.done:
            self.app.random_index = self.app.random_index + 1#random.randint(0, 124)
            self.app.randombox = self.app.box_creator.preview(125)[self.app.random_index][:-1]
            self.run_step()

import os
os.environ['ETS_TOOLKIT'] = 'qt4'
from pyface.qt import QtGui, QtCore
from traits.api import HasTraits, Instance, on_trait_change
from traitsui.api import View, Item, HSplit, VSplit, InstanceEditor, HGroup
from mayavi.core.api import Engine
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor


################################################################################
#The actual visualization
class Visualization(HasTraits):
    scene = Instance(MlabSceneModel, ())
    scene1 = Instance(MlabSceneModel, ())
    
    @on_trait_change('scene.activated')
    def update_plot(self):
        print('r')

    view = View(HGroup(Item('scene', editor=SceneEditor(scene_class=MayaviScene), width=480, height=480, show_label=False),
                            Item('scene1', editor=SceneEditor(scene_class=MayaviScene), width=480, height=480, show_label=False)),
                resizable=True)#, scrollable=True)

class Visualization_1box(HasTraits):
    scene3 = Instance(MlabSceneModel, ())
    
    @on_trait_change('scene3.activated')
    def update_plot(self):
        print('r1')

    view = View(Item('scene3', editor=SceneEditor(scene_class=MayaviScene), width=480, height=480, show_label=False),
                resizable=True)#, scrollable=True)
    
################################################################################
# The QWidget containing the visualization, this is pure PyQt4 code.
class MayaviQWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        layout = QtGui.QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        self.visualization = Visualization()

        self.ui = self.visualization.edit_traits(parent=self, kind='subpanel').control
        layout.addWidget(self.ui)
        self.ui.setParent(self)

class window(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # CON_STR = {
        #     dType.DobotConnect.DobotConnect_NoError:  "DobotConnect_NoError",
        #     dType.DobotConnect.DobotConnect_NotFound: "DobotConnect_NotFound",
        #     dType.DobotConnect.DobotConnect_Occupied: "DobotConnect_Occupied"}
        
        # self.api = dType.load()
        # #threading.Thread(target=self.load_model_envs).start()
        # self.state = dType.ConnectDobot(self.api, "COM3", 115200)[0]
        # print("Connect status:",CON_STR[self.state])
        # dType.SetQueuedCmdClear(self.api)
        
        uic.loadUi("pygui//mainwindow.ui", self)
        self.i = 0

        self.setWindowTitle("Graph User Interface")
        self.setWindowIcon(QtGui.QIcon("image\\box.png"))
        
        layout = QtGui.QVBoxLayout(self.widget)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        self.widget.visualization = Visualization()
        self.widget.ui = self.widget.visualization.edit_traits(parent=self.widget, kind='subpanel').control
        layout.addWidget(self.widget.ui)
        self.widget.ui.setParent(self.widget)

        self.size_con = (10, 10, 10)#(61, 61, 61)
        self.draw_container_widget(self.widget.visualization.scene.mayavi_scene)
        self.draw_container_widget(self.widget.visualization.scene1.mayavi_scene)

        import cv2
        img = cv2.imread('image\\box.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        scale_percent = 17 # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        h, w, ch = img.shape
        bytes_per_line = ch * w
        qimage = QtGui.QImage(img.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        self.label_box_new.setPixmap(QtGui.QPixmap.fromImage(qimage))

        self.pushButton_run_one_shot.setIcon(QIcon("image\\icon_play_one_shot.png"))
        self.pushButton_run_one_shot.setIconSize(QtCore.QSize(61,61))
        self.pushButton_run_one_shot.clicked.connect(lambda: self.run_one_shot())
        self.pushButton_play.setIcon(QIcon("image\\icon_play.png"))
        self.pushButton_play.setIconSize(QtCore.QSize(61,61))
        self.pushButton_play.clicked.connect(lambda: self.play())
        self.pushButton_reset.setIcon(QIcon("image\\icon_reset.png"))
        self.pushButton_reset.setIconSize(QtCore.QSize(61,61))
        self.pushButton_reset.clicked.connect(lambda: self.reset())

        self.pushButton_add_custom_box.clicked.connect(lambda: self.custom_box())
        self.pushButton_add_random_box.clicked.connect(lambda: self.random_box())
        self.pushButton_add_random_box.setStyleSheet("background-color : yellow")
        
        self.pushButton_load_env_model.clicked.connect(lambda: threading.Thread(target=self.load_model_envs).start())
        #self.pushButton_load_env_model.clicked.connect(lambda: self.load_model_envs())

        #############################################################
        ############### tab actual run ##############################
        #############################################################
        layout_actual = QtGui.QVBoxLayout(self.widget_simulate)
        layout_actual.setContentsMargins(0,0,0,0)
        layout_actual.setSpacing(0)
        self.widget_simulate.visualization = Visualization_1box()
        self.widget_simulate.ui = self.widget_simulate.visualization.edit_traits(parent=self.widget_simulate, kind='subpanel').control
        layout_actual.addWidget(self.widget_simulate.ui)
        self.widget_simulate.ui.setParent(self.widget_simulate)
        

        #self.size_con = (61, 61, 61)
        self.draw_container_widget(self.widget_simulate.visualization.scene3.mayavi_scene)

        self.label_box_new_actual.setPixmap(QtGui.QPixmap.fromImage(qimage))
        self.pushButton_load_env_model_actual.clicked.connect(lambda: threading.Thread(target=self.load_env_model_actual).start())
        self.pushButton_connect_camera.clicked.connect(lambda: threading.Thread(target=self.connect_camera).start())

        self.pushButton_run_one_shot_actual.setIcon(QIcon("image\\icon_play_one_shot.png"))
        self.pushButton_run_one_shot_actual.setIconSize(QtCore.QSize(61,61))
        self.pushButton_run_one_shot_actual.clicked.connect(lambda: self.run_actual_one_shot())
        self.pushButton_play_actual.setIcon(QIcon("image\\icon_play.png"))
        self.pushButton_play_actual.setIconSize(QtCore.QSize(61,61))
        #self.pushButton_play_actual.clicked.connect(lambda: self.play())
        self.pushButton_reset_actual.setIcon(QIcon("image\\icon_reset.png"))
        self.pushButton_reset_actual.setIconSize(QtCore.QSize(61,61))
        #self.pushButton_reset_actual.clicked.connect(lambda: self.reset())

        self.cap = cv2.VideoCapture(1)
        self.box_creator = LoadBoxCreator('PCB-dataset\\setting123_discrete.pt')
        self.box_creator.reset()
        
        self.random_index = random.randint(0, 124)
        self.randombox = self.box_creator.preview(125)[self.random_index][:-1]
        
        self.pushButton_test_robot.clicked.connect(lambda: threading.Thread(target=self.test_robot).start())

    #Delays commands
    def commandDelay(self, lastIndex):
        dType.SetQueuedCmdStartExec(self.api)
        self.label_packed.setStyleSheet("background-color : yellow")
        while lastIndex > dType.GetQueuedCmdCurrentIndex(self.api)[0]:
            dType.dSleep(200)

        if lastIndex == dType.GetQueuedCmdCurrentIndex(self.api)[0]:
            self.label_packed.setStyleSheet("background-color : green")
        dType.SetQueuedCmdStopExec(self.api)

    def test_robot(self):
        dType.SetHOMECmd(self.api, temp = 0, isQueued = 1)
        lastIndex = dType.SetPTPCmd(self.api, dType.PTPMode.PTPMOVLXYZMode, 224, -172, 124, -8, isQueued = 1)[0]
        self.commandDelay(lastIndex)
        #self.pick_and_place_item([0, 0, 0])

    def pick_and_place_item(self, toa_do):
        def pick(x, y, z, r):
            global dType, lastIndex
            lastIndex = dType.SetPTPCmd(self.api, dType.PTPMode.PTPMOVLXYZMode, x, y, z + 100, r, isQueued = 1)[0]
            lastIndex = dType.SetPTPCmd(self.api, dType.PTPMode.PTPMOVLXYZMode, x, y, z, r, isQueued = 1)[0]
            lastIndex = dType.SetEndEffectorSuctionCup(self.api, True, True, isQueued = 1)[0]
            dType.dSleep(30)
            lastIndex = dType.SetPTPCmd(self.api, dType.PTPMode.PTPMOVLXYZMode, x, y, z + 100, r, isQueued = 1)[0]

        def place(x, y, z, r):
            global dType, lastIndex
            lastIndex = dType.SetPTPCmd(self.api, dType.PTPMode.PTPMOVLXYZMode, x, y, z+100, r, isQueued = 1)[0]#vi tri dat 1
            lastIndex = dType.SetPTPCmd(self.api, dType.PTPMode.PTPMOVLXYZMode, x, y, z, r, isQueued = 1)[0]#vi tri dat 1
            lastIndex = dType.SetEndEffectorSuctionCup(self.api, True, False, isQueued = 1)[0]
            dType.dSleep(30)
            lastIndex = dType.SetPTPCmd(self.api, dType.PTPMode.PTPMOVLXYZMode, x, y, z+100, r, isQueued = 1)[0]

        def convert(x, y, z, xoay = False, kk = 20):
            r = -8
            hs = 11
            x, y, z = 70 + x*hs, -260 + y*10.5, -49 + z*10
            return x, y, z, r
                
        x, y, z, r = -30, -260, -49, -8
        pick(x, y, z, r)
        x, y, z, r = convert(toa_do[0], toa_do[1], toa_do[2], xoay = False)
        place(x, y, z, r)
        self.commandDelay(lastIndex)

    def run_actual_one_shot(self, actual_next_box):
        update_1_a = False
        color_box = tuple(np.random.rand(1, 3)[0])
        self.model_actual.envs.type_gen_box = "Actual"
        self.model_actual.envs.next_box_cf = actual_next_box
        self.model_actual.run_step()
        if self.i_a < len(self.model_actual.box):
            self.draw_actual_box2container(self.widget_simulate.visualization.scene3.mayavi_scene, self.model_actual.box[self.i_a], color_box)
            update_1_a = True
        self.i_a = self.i_a + 1
        self.update_state_actual_container(update_1_a)
    
    def update_state_actual_container(self, update_1_a):
        if update_1_a:
            self.label_ratio_V_actual.setText("Ratio V: " + str(self.model_actual.envs.space.get_ratio()))
            self.label_num_of_box_actual.setText("Num of box: " + str(len(self.model_actual.envs.space.boxes)))

    def load_env_model_actual(self):
        self.i_a = 0
        if self.comboBox_select_model_actual.currentText() == "Model RL":
            self.model_actual = run_rl_model(self, stt_model=3)

    def connect_camera(self):
        #self.label_image_data_matrix.resize(500, 280)
        print("connected")
        t = True
        while t:
            _, frame = self.cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = gray[250:500, 250:500]
            blur = cv.GaussianBlur(gray,(5,5),0)
            ret,thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            msg = pylibdmtx.decode(thresh)
            print(msg)
            dim = (500, 500)
            frame = frame[250:500, 250:500, :]
            frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qimage = QtGui.QImage(frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            self.label_image_data_matrix.setPixmap(QtGui.QPixmap.fromImage(qimage))

            if len(msg) != 0:
                msg = msg[0].data
                msg = msg.decode("utf-8").split('x')
                if len(msg) == 4:
                    self.lineEdit_X_actual.setText(str(msg[1]))
                    self.lineEdit_Y_actual.setText(str(msg[2]))
                    self.lineEdit_Z_actual.setText(str(msg[3]))
                    self.label_content_datamatrix.setText("Content:" + str(msg))
                    self.run_actual_one_shot((int(msg[1]), int(msg[2]), int(msg[3])))
            t = False

    def custom_box(self):
        self.pushButton_add_custom_box.setStyleSheet("background-color : yellow")
        self.pushButton_add_random_box.setStyleSheet("background-color : white")
        if self.model1.envs.type_gen_box == "Auto":
            self.model1.envs.type_gen_box = "Manual"
            self.model2.envs.type_gen_box = "Manual"

    def random_box(self):
        self.pushButton_add_custom_box.setStyleSheet("background-color : white")
        self.pushButton_add_random_box.setStyleSheet("background-color : yellow")
        if self.model1.envs.type_gen_box == "Manual":
            self.model1.envs.type_gen_box = "Auto"
            self.model2.envs.type_gen_box = "Auto"

    def run_one_shot(self):
        update_1 = False
        update_2 = False
        color_box = tuple(np.random.rand(1, 3)[0])
        self.random_index = self.random_index + 1 
        self.randombox = self.box_creator.preview(125)[self.random_index][:-1]
        self.model1.run_step()
        if self.i < len(self.model1.box):
            self.draw_box2container(self.widget.visualization.scene.mayavi_scene, self.model1.box[self.i], color_box)
            update_1 = True
        self.model2.run_step()
        if self.i < len(self.model2.box):
            self.draw_box2container(self.widget.visualization.scene1.mayavi_scene, self.model2.box[self.i], color_box)
            update_2 = True
        self.i = self.i+1
        self.update_state_container(update_1, update_2)

    def play(self):
        # self.model1.run_complete()
        # self.model2.run_complete()
        
        while not self.model1.done and not self.model2.done:
            self.random_index = random.randint(0, 124)
            self.randombox = self.box_creator.preview(125)[self.random_index][:-1]
            if not self.model1.done:
                self.model1.run_step()
            if not self.model2.done:
                self.model2.run_step()
        
        for i in range(self.i, max(len(self.model1.box), len(self.model2.box))):
            color_box = tuple(np.random.rand(1, 3)[0])
            if i < len(self.model1.box):
                self.draw_box2container(self.widget.visualization.scene.mayavi_scene, self.model1.envs.space.boxes[i], color_box)
            if i < len(self.model2.box):
                self.draw_box2container(self.widget.visualization.scene1.mayavi_scene, self.model2.envs.space.boxes[i], color_box)
        
        self.update_state_container(True, True)
        

    def reset(self):
        self.random_box()
        self.widget.visualization.scene.mlab.clf(figure = self.widget.visualization.scene.mayavi_scene)
        self.widget.visualization.scene.mlab.clf(figure = self.widget.visualization.scene1.mayavi_scene)
        self.i = 0
        self.label_ratio_V_1.setText("Ratio V: ")
        self.label_num_of_box_1.setText("Num of box: ")
        self.label_ratio_V_2.setText("Ratio V: ")
        self.label_num_of_box_2.setText("Num of box: ")
        self.draw_container_widget(self.widget.visualization.scene.mayavi_scene)
        self.draw_container_widget(self.widget.visualization.scene1.mayavi_scene)
        self.load_model_envs()

    def load_model_envs(self):
        if self.comboBox_select_model_1.currentText() == "Model RL":
            self.model1 = run_rl_model(self, stt_model=1)
        if self.comboBox_select_model_2.currentText() == "Model RL":
            self.model2 = run_rl_model(self, stt_model=2)

    def update_state_container(self, update_1, update_2):
        if update_1:
            self.label_ratio_V_1.setText("Ratio V: " + str(self.model1.envs.space.get_ratio()))
            self.label_num_of_box_1.setText("Num of box: " + str(len(self.model1.envs.space.boxes)))
        if update_2:
            self.label_ratio_V_2.setText("Ratio V: " + str(self.model2.envs.space.get_ratio()))
            self.label_num_of_box_2.setText("Num of box: " + str(len(self.model2.envs.space.boxes)))

    def draw_container_widget(self, name):
        size_con = self.size_con[0]
        X = np.zeros([3]); Y = np.zeros([3]); Z = np.mgrid[0 : size_con : 3j]
        self.widget.visualization.scene.mlab.plot3d(X, Y, Z, color = (0, 0, 0), line_width = 3, figure=name)
        self.widget.visualization.scene.mlab.plot3d(X, Y+size_con, Z, color = (0, 0, 0), line_width = 3)
        self.widget.visualization.scene.mlab.plot3d(X+size_con, Y, Z, color = (0, 0, 0), line_width = 3)
        self.widget.visualization.scene.mlab.plot3d(X+size_con, Y+size_con, Z, color = (0, 0, 0), line_width = 3)
        
        X = np.zeros([3]); Z = np.zeros([3]); Y = np.mgrid[0 : size_con : 3j]
        self.widget.visualization.scene.mlab.plot3d(X, Y, Z, color = (0, 0, 0), line_width = 3)
        self.widget.visualization.scene.mlab.plot3d(X+size_con, Y, Z, color = (0, 0, 0), line_width = 3)
        self.widget.visualization.scene.mlab.plot3d(X+size_con, Y, Z+size_con, color = (0, 0, 0), line_width = 3)
        self.widget.visualization.scene.mlab.plot3d(X, Y, Z+size_con, color = (0, 0, 0), line_width = 3)

        Z = np.zeros([3]); Y = np.zeros([3]); X = np.mgrid[0 : size_con : 3j]
        self.widget.visualization.scene.mlab.plot3d(X, Y, Z, color = (0, 0, 0), line_width = 3)
        self.widget.visualization.scene.mlab.plot3d(X, Y+size_con, Z, color = (0, 0, 0), line_width = 3)
        self.widget.visualization.scene.mlab.plot3d(X, Y, Z+size_con, color = (0, 0, 0), line_width = 3)
        self.widget.visualization.scene.mlab.plot3d(X, Y+size_con, Z+size_con, color = (0, 0, 0), line_width = 3)

    def draw_container(self, name):
        #self.mayavi_widget.visualization.scene.mlab.test_points3d()
        #self.mayavi_widget.visualization.scene.mlab.clf()
        X = np.zeros([3]); Y = np.zeros([3]); Z = np.mgrid[0 : 10 : 3j]
        self.mayavi_widget.visualization.scene.mlab.plot3d(X, Y, Z, color = (0, 0, 0), line_width =3, figure=name)
        self.mayavi_widget.visualization.scene.mlab.plot3d(X, Y+10, Z, color = (0, 0, 0), line_width =3)
        self.mayavi_widget.visualization.scene.mlab.plot3d(X+10, Y, Z, color = (0, 0, 0), line_width =3)
        self.mayavi_widget.visualization.scene.mlab.plot3d(X+10, Y+10, Z, color = (0, 0, 0), line_width =3)
        
        X = np.zeros([3]); Z = np.zeros([3]); Y = np.mgrid[0 : 10 : 3j]
        self.mayavi_widget.visualization.scene.mlab.plot3d(X, Y, Z, color = (0, 0, 0), line_width =3)
        self.mayavi_widget.visualization.scene.mlab.plot3d(X+10, Y, Z, color = (0, 0, 0), line_width =3)
        self.mayavi_widget.visualization.scene.mlab.plot3d(X+10, Y, Z+10, color = (0, 0, 0), line_width =3)
        self.mayavi_widget.visualization.scene.mlab.plot3d(X, Y, Z+10, color = (0, 0, 0), line_width =3)
        Z = np.zeros([3]); Y = np.zeros([3]); X = np.mgrid[0 : 10 : 3j]
        self.mayavi_widget.visualization.scene.mlab.plot3d(X, Y, Z, color = (0, 0, 0), line_width =3)
        self.mayavi_widget.visualization.scene.mlab.plot3d(X, Y+10, Z, color = (0, 0, 0), line_width =3)
        self.mayavi_widget.visualization.scene.mlab.plot3d(X, Y, Z+10, color = (0, 0, 0), line_width =3)
        self.mayavi_widget.visualization.scene.mlab.plot3d(X, Y+10, Z+10, color = (0, 0, 0), line_width =3)
        # self.mayavi_widget.visualization.scene.parallel_projection = True
        # self.mayavi_widget.visualization.scene.isometric_view = True
        # self.mayavi_widget.visualization.scene.camera.azimuth(5)
        # self.mayavi_widget.visualization.scene.mlab.move(1, -1, -1)

    def model_run_by_step(self):
        update_1 = False
        update_2 = False
        color_box = tuple(np.random.rand(1, 3)[0])
        self.model1.run_step()
        if self.i < len(self.model1.box):
            self.draw_box2container(self.mayavi_widget.visualization.scene.mayavi_scene, self.model1.box[self.i], color_box)
            update_1 = True
        self.model2.run_step()
        if self.i < len(self.model2.box):
            self.draw_box2container(self.mayavi_widget.visualization.scene1.mayavi_scene, self.model2.box[self.i], color_box)
            update_2 = True
        self.i = self.i+1
        self.update_state_container(update_1, update_2)

    def draw_actual_box2container(self, name, box, color_box):
        x, y, z = box.lx, box.ly, box.lz
        dx,dy,dz = box.x, box.y, box.z
        Y, Z= np.mgrid[y : y+dy : 3j, z : z+dz : 3j]
        X = np.ones([3, 3])*x
        self.widget_simulate.visualization.scene3.mlab.mesh(X, Y, Z, color=color_box, figure = name)
        self.widget_simulate.visualization.scene3.mlab.mesh(X+dx, Y, Z, color=color_box)

        Y, X= np.mgrid[y : y+dy : 3j, x : x+dx : 3j]
        Z = np.ones([3, 3])*z
        self.widget_simulate.visualization.scene3.mlab.mesh(X, Y, Z, color=color_box)
        self.widget_simulate.visualization.scene3.mlab.mesh(X, Y, Z+dz, color=color_box)

        Z, X= np.mgrid[z : z+dz : 3j, x : x+dx : 3j]
        Y = np.ones([3, 3])*y
        self.widget_simulate.visualization.scene3.mlab.mesh(X, Y, Z, color=color_box)
        self.widget_simulate.visualization.scene3.mlab.mesh(X, Y+dy, Z, color=color_box)

    def draw_box2container(self, name, box, color_box):
        x, y, z = box.lx, box.ly, box.lz
        dx,dy,dz = box.x, box.y, box.z
        Y, Z= np.mgrid[y : y+dy : 3j, z : z+dz : 3j]
        X = np.ones([3, 3])*x
        self.widget.visualization.scene.mlab.mesh(X, Y, Z, color=color_box, figure = name)
        self.widget.visualization.scene.mlab.mesh(X+dx, Y, Z, color=color_box)

        Y, X= np.mgrid[y : y+dy : 3j, x : x+dx : 3j]
        Z = np.ones([3, 3])*z
        self.widget.visualization.scene.mlab.mesh(X, Y, Z, color=color_box)
        self.widget.visualization.scene.mlab.mesh(X, Y, Z+dz, color=color_box)

        Z, X= np.mgrid[z : z+dz : 3j, x : x+dx : 3j]
        Y = np.ones([3, 3])*y
        self.widget.visualization.scene.mlab.mesh(X, Y, Z, color=color_box)
        self.widget.visualization.scene.mlab.mesh(X, Y+dy, Z, color=color_box)

    def test_ani(self):
        for box in self.box:
            x, y, z = box.lx, box.ly, box.lz
            dx,dy,dz = box.x, box.y, box.z
            #print(x, y, z, dx, dy, dz)
            color_box = tuple(np.random.rand(1, 3)[0])
            #x = 10; y = 10; z = 10; dx = 10; dy = 10; dz = 10
            Y, Z= np.mgrid[y : y+dy : 3j, z : z+dz : 3j]
            X = np.ones([3, 3])*x
            self.mayavi_widget.visualization.scene.mlab.mesh(X, Y, Z, color=color_box)
            self.mayavi_widget.visualization.scene.mlab.mesh(X+dx, Y, Z, color=color_box)

            Y, X= np.mgrid[y : y+dy : 3j, x : x+dx : 3j]
            Z = np.ones([3, 3])*z
            self.mayavi_widget.visualization.scene.mlab.mesh(X, Y, Z, color=color_box)
            self.mayavi_widget.visualization.scene.mlab.mesh(X, Y, Z+dz, color=color_box)

            Z, X= np.mgrid[z : z+dz : 3j, x : x+dx : 3j]
            Y = np.ones([3, 3])*y
            self.mayavi_widget.visualization.scene.mlab.mesh(X, Y, Z, color=color_box)
            self.mayavi_widget.visualization.scene.mlab.mesh(X, Y+dy, Z, color=color_box)

def main():
    app = QtWidgets.QApplication(sys.argv)
    ex = window()
    ex.showMaximized()
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

