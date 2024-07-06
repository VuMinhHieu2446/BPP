import numpy as np
from mayavi.mlab import *
import mayavi.mlab as mlab
import sys
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5 import uic
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import time
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
# args_id = 'PctContinuous-v0'
# args_id = 'PctDiscrete-v0'
# args_setting = 2
# if args_setting == 1:
#     args_internal_node_length = 6
# elif args_setting == 2:
#     args_internal_node_length = 6
# elif args_setting == 3:
#     args_internal_node_length = 7
# args_container_size = givenData.container_size
# args_item_size_set  = givenData.item_size_set
# args_dataset_path = "D:/New folder (5)/20212/do-an-20221/New folder (4)/PCB-dataset/setting123_discrete.pt"
# #args_dataset_path = "D:/New folder (5)/20212/do-an-20221/New folder (4)/PCB-dataset/setting2_continuous.pt"
# #args_dataset_path = "D:/New folder (5)/20212/do-an-20221/New folder (4)/PCB-dataset/setting13_continuous.pt"
# args_load_dataset = True
# args_internal_node_holder = 150 #Maximum number of internal nodes
# args_leaf_node_holder = 50 #Maximum number of leaf nodes
# args_lnes = 'EMS' #Leaf Node Expansion Schemes: EMS (recommend), EV, EP, CP, FC
# args_shuffle = True #Randomly shuffle the leaf nodes
# args_num_processes = 1 #The number of parallel processes used for training
# args_device = 'cpu'
# device = args_device
# envs = gym.make(args_id,
#                 setting=args_setting,
#                 container_size=args_container_size,
#                 item_set=args_item_size_set,
#                 data_name=args_dataset_path,
#                 load_test_data=args_load_dataset,
#                 internal_node_holder=args_internal_node_holder,
#                 leaf_node_holder=args_leaf_node_holder,
#                 LNES=args_lnes,
#                 shuffle=args_shuffle)

# args_embedding_size = 64 #Dimension of input embedding
# args_hidden_size = 128
# args_gat_layer_num = 1
# PCT_policy =  DRL_GAT(args_embedding_size,
#                  args_hidden_size,
#                  args_gat_layer_num,
#                  args_internal_node_holder,
#                  args_internal_node_length,
#                  args_leaf_node_holder)

# args_model_path = 'D:/New folder (5)/20212/do-an-20221/New folder/PCB-model/setting2_discrete.pt'
# args_model_path = "D:/New folder (5)/20212/do-an-20221/New folder (4)/PCB-model/PCT-model3_tung-2022.11.22-20-40-06_2022.11.22-20-41-14.pt"
# PCT_policy = load_policy(args_model_path, PCT_policy)
# print('Pre-train model loaded!')

class run_model():
    def __init__(self, model_path):
        self.id = 'PctContinuous-v0'
        self.id = 'PctDiscrete-v0'
        self.setting = 2
        if self.setting == 1:
            self.internal_node_length = 6
        elif self.setting == 2:
            self.internal_node_length = 6
        elif self.setting == 3:
            self.internal_node_length = 7
        self.container_size = givenData.container_size
        self.item_size_set  = givenData.item_size_set
        self.dataset_path = "D:/New folder (5)/20212/do-an-20221/New folder (4)/PCB-dataset/setting123_discrete.pt"
        self.load_dataset = True
        self.internal_node_holder = 80 #Maximum number of internal nodes
        self.leaf_node_holder = 50 #Maximum number of leaf nodes
        self.lnes = 'EMS' #Leaf Node Expansion Schemes: EMS (recommend), EV, EP, CP, FC
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

        self.model_path = "D:/New folder (5)/20212/do-an-20221/all_model/PCT-TUNG-2023.02.04-07-07-37_2023.02.04-07-08-46.pt"
        self.model_path = model_path
        #'D:/New folder (5)/20212/do-an-20221/New folder/PCB-model/setting2_discrete.pt'
        #"D:/New folder (5)/20212/do-an-20221/New folder (4)/PCB-model/PCT-model3_tung-2022.11.22-20-40-06_2022.11.22-20-41-14.pt"
        self.PCT_policy = load_policy(self.model_path, self.PCT_policy)
        print('Pre-train model loaded!')
        self.envs.type_gen_box = "Manual"
        self.envs.type_gen_box = "Auto"
        self.PCT_policy.eval()
        self.envs.next_box_cf = (5, 5, 5)#first box
        self.obs = self.envs.reset()
        self.obs = torch.FloatTensor(self.obs).to(self.device).unsqueeze(dim=0)
        self.all_nodes, self.leaf_nodes = tools.get_leaf_nodes_with_factor(self.obs, self.num_processes, self.internal_node_holder, self.leaf_node_holder)

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

    def load_model_and_envs(self):
        pass
    
    def run_step(self, *arg):
        with torch.no_grad():
            selectedlogProb, selectedIdx, policy_dist_entropy, value = self.PCT_policy(self.all_nodes, True, normFactor = self.factor)
        selected_leaf_node = self.leaf_nodes[self.batchX, selectedIdx.squeeze()]
        self.items = self.envs.packed
            
        self.obs, reward, done, infos = self.envs.step(selected_leaf_node.cpu().numpy()[0][0:6], (5, 5, 5))#, my_next_box)
        self.box = self.envs.space.boxes#[:-1]
        # frames.append(envs.render(mode='rgb_array'))
        # print(len(envs.space.boxes[:-1]))
        # for box in envs.space.boxes[:-1]:
        #     x, y, z = box.lx, box.ly, box.lz
        #     dx,dy,dz = box.x, box.y, box.z
        #     print(x, y, z, dx, dy, dz)

        if done:
            print('Episode {} ends.'.format(self.step_counter))
            if 'ratio' in infos.keys():
                self.episode_ratio.append(infos['ratio'])
            if 'counter' in infos.keys():
                self.episode_length.append(infos['counter'])

            print('Mean ratio: {}, length: {}'.format(np.mean(self.episode_ratio), np.mean(self.episode_length)))
            print('Episode ratio: {}, length: {}'.format(infos['ratio'], infos['counter']))
            self.all_episodes.append(self.items)
            self.step_counter += 1
            self.obs = self.envs.reset()

        self.obs = torch.FloatTensor(self.obs).to(self.device).unsqueeze(dim=0)
        self.all_nodes, self.leaf_nodes = tools.get_leaf_nodes_with_factor(self.obs, self.num_processes, self.internal_node_holder, self.leaf_node_holder)               

    def run_complete(self):
        while self.step_counter < self.eval_freq:
            self.run_step()

# First, and before importing any Enthought packages, set the ETS_TOOLKIT
# environment variable to qt4, to tell Traits that we will use Qt.
import os
os.environ['ETS_TOOLKIT'] = 'qt4'
# By default, the PySide binding will be used. If you want the PyQt bindings
# to be used, you need to set the QT_API environment variable to 'pyqt'
#os.environ['QT_API'] = 'pyqt'

# To be able to use PySide or PyQt4 and not run in conflicts with traits,
# we need to import QtGui and QtCore from pyface.qt
from pyface.qt import QtGui, QtCore
# Alternatively, you can bypass this line, but you need to make sure that
# the following lines are executed before the import of PyQT:
#   import sip
#   sip.setapi('QString', 2)

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
        # This function is called when the view is opened. We don't
        # populate the scene when the view is not yet open, as some
        # VTK features require a GLContext.

        # We can do normal mlab calls on the embedded scene.
        #self.scene.isometric_view()
        print('r')


    # the layout of the dialog screated
    # view = View(Item("scene", editor=SceneEditor(scene_class=MayaviScene),
    #                  height=250, width=300, show_label=False),
    #             resizable=True # We need this to resize with the parent widget
    #             )
    view = View(HGroup(Item('scene', editor=SceneEditor(scene_class=MayaviScene), width=480, height=480, show_label=False),
                            Item('scene1', editor=SceneEditor(scene_class=MayaviScene), width=480, height=480, show_label=False)),
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

        # If you want to debug, beware that you need to remove the Qt
        # input hook.
        #QtCore.pyqtRemoveInputHook()
        #import pdb ; pdb.set_trace()
        #QtCore.pyqtRestoreInputHook()

        # The edit_traits call will generate the widget to embed.
        self.ui = self.visualization.edit_traits(parent=self, kind='subpanel').control
        layout.addWidget(self.ui)
        self.ui.setParent(self)

class window(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.resize(200, 200)
        uic.loadUi("C://Users//HP//Desktop//mainwindow.ui", self)
        # self.setGeometry(100, 100, 600, 400) set kich thuoc window

        #self.model1 = run_model("D:/New folder (5)/20212/do-an-20221/New folder (4)/PCB-model/PCT-model3_tung-2022.11.22-20-40-06_2022.11.22-20-41-14.pt")
        #"D:\New folder (5)\20212\do-an-20221\all_model\num1_setting_1_lnes_EMS_use_acktr_true_400_step.pt"
        # self.model1 = run_model("D:/New folder (5)/20212/do-an-20221/all_model/num2_setting_2_lnes_EMS_use_acktr_true.pt")
        # self.model1.envs.type_gen_box = "Auto"
        # #self.model1.run_complete()
        # self.model2 = run_model("D:/New folder (5)/20212/do-an-20221/New folder/PCB-model/setting2_discrete.pt")
        # self.model2.envs.type_gen_box = "Auto"

        # self.setWindowTitle("PyQt5")
        # self.setWindowIcon(QtGui.QIcon("D:\\New folder (5)\\20212\\do-an-20221\\image\\box.png"))
        #self.run_model_full()

        # self.mayavi_widget = MayaviQWidget(self)
        # self.mayavi_widget.resize(1000, 500)
        # self.mayavi_widget.move(0,0)
        # self.draw_container(self.mayavi_widget.visualization.scene.mayavi_scene)
        # self.draw_container(self.mayavi_widget.visualization.scene1.mayavi_scene)

        
        layout = QtGui.QVBoxLayout(self.widget)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        self.widget.visualization = Visualization()
        self.widget.ui = self.widget.visualization.edit_traits(parent=self.widget, kind='subpanel').control
        layout.addWidget(self.widget.ui)
        self.widget.ui.setParent(self.widget)

        self.draw_container_widget(self.widget.visualization.scene.mayavi_scene)
        self.draw_container_widget(self.widget.visualization.scene1.mayavi_scene)


        # self.label_1 = QLabel("Model 1: ", self)# creating a label widget
        # self.label_1.move(70, 520)# moving position
        # self.label_1.resize(200, 30)
        # self.label_1.setStyleSheet("font: bold 24px")# setting up border
        # self.label_ratio_1 = QLabel("Tỉ lệ: ", self)# creating a label widget
        # self.label_ratio_1.move(130, 570)# moving position
        # self.label_ratio_1.resize(200, 30)
        # self.label_ratio_1.setStyleSheet("font: bold 24px")# setting up border
        # self.label_num_1 = QLabel("Số lượng: ", self)# creating a label widget
        # self.label_num_1.move(130, 620)# moving position
        # self.label_num_1.resize(200, 30)
        # self.label_num_1.setStyleSheet("font: bold 24px")# setting up border

        # self.label_2 = QLabel("Model 2: ", self)# creating a label widget
        # self.label_2.move(600, 520)# moving position
        # self.label_2.resize(200, 30)
        # self.label_2.setStyleSheet("font: bold 24px")# setting up border
        # self.label_ratio_2 = QLabel("Tỉ lệ: ", self)# creating a label widget
        # self.label_ratio_2.move(650, 570)# moving position
        # self.label_ratio_2.resize(150, 30)
        # self.label_ratio_2.setStyleSheet("font: bold 24px")# setting up border
        # self.label_num_2 = QLabel("Số lượng: ", self)# creating a label widget
        # self.label_num_2.move(650, 620)# moving position
        # self.label_num_2.resize(150, 30)
        # self.label_num_2.setStyleSheet("font: bold 24px")# setting up border

        #self.image_confi_box = QtWidgets.QLabel()

        # self.button_test_mayavi = QtWidgets.QPushButton(self)
        # self.button_test_mayavi.setText("Test Mayavi")
        # self.button_test_mayavi.move(1000, 5)
        # self.i = 0
        # self.button_test_mayavi.clicked.connect(lambda: self.model_run_by_step())

        # self.button_test_mayavi2 = QtWidgets.QPushButton(self)
        # self.button_test_mayavi2.setText("Test Mayavi 2")
        # self.button_test_mayavi2.move(1000, 70)
        # self.button_test_mayavi2.clicked.connect(lambda: self.model_run_by_step())

        #set envs and PCT
        # self.label_load_model_envs = QLabel(self)
        # self.label_load_model_envs.setText("Cài đặt môi trường và model")
        # self.label_load_model_envs.move(1020, 20)

        # self.cb_lnes = QComboBox(self)
        # self.cb_lnes.addItem("Extreme Point")
        # self.cb_lnes.addItem("Empty Maximal Space")
        # self.cb_lnes.addItem("Event Point")
        # self.cb_lnes.move(1040, 50)
        # #print(self.cb_lnes.currentText())
        # #self.cb_lnes.currentIndexChanged.connect(self.selectionchange)

        # self.cb_setting = QComboBox(self)
        # self.cb_setting.addItem("Setting = 1")
        # self.cb_setting.addItem("Setting = 2")
        # self.cb_setting.move(1040, 80)

        # self.btn_load_model_envs = QPushButton(self)
        # self.btn_load_model_envs.setText("Khởi tạo môi trường và model")
        # self.btn_load_model_envs.move(1040, 110)
        # self.btn_load_model_envs.clicked.connect(lambda: self.load_model_envs())

        # #add new box
        # self.label_add_new_box = QLabel(self)
        # self.label_add_new_box.setText("Thêm hộp mới")
        # self.label_add_new_box.move(1020, 140)
        # import cv2
        # img = cv2.imread('D:\\New folder (5)\\20212\\do-an-20221\\image\\box.png')
        # #"D:\New folder (5)\20212\do-an-20221\image\box.png"
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # scale_percent = 20 # percent of original size
        # width = int(img.shape[1] * scale_percent / 100)
        # height = int(img.shape[0] * scale_percent / 100)
        # dim = (width, height)
        # img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        # h, w, ch = img.shape
        # bytes_per_line = ch * w
        # qimage = QtGui.QImage(img.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        # self.label_box = QtWidgets.QLabel(self)
        # self.label_box.setText("Hello World")
        # self.label_box.move(1200, 170)
        # self.label_box.resize(w, h)
        # self.label_box.setPixmap(QtGui.QPixmap.fromImage(qimage))

        # self.line_X = QLineEdit(self)
        # self.line_X.setText("X (mm):")
        # self.line_X.resize(100, 30)
        # self.line_X.move(1040, 170)
        # self.line_Y = QLineEdit(self)
        # self.line_Y.setText("Y (mm):")
        # self.line_Y.resize(100, 30)
        # self.line_Y.move(1040, 210)
        # self.line_Z = QLineEdit(self)
        # self.line_Z.setText("Z (mm):")
        # self.line_Z.resize(100, 30)
        # self.line_Z.move(1040, 250)

        # self.btn_add_edited_box = QPushButton(self)
        # self.btn_add_edited_box.setText("Add edited box")
        # self.btn_add_edited_box.move(1040, 400)

        # self.btn_add_random_box = QPushButton(self)
        # self.btn_add_random_box.setText("Add random box")
        # self.btn_add_random_box.move(1140, 400)
        
    def load_model_envs(self):
        pass

    def update_state_container(self, update_1, update_2):
        if update_1:
            self.label_ratio_1.setText("Tỉ lệ: " + str(self.model1.envs.space.get_ratio()))
            self.label_num_1.setText("Số lượng: " + str(len(self.model1.envs.space.boxes)))
        if update_2:
            self.label_ratio_2.setText("Tỉ lệ: " + str(self.model2.envs.space.get_ratio()))
            self.label_num_2.setText("Số lượng: " + str(len(self.model2.envs.space.boxes)))

    def draw_container_widget(self, name):
        #self.mayavi_widget.visualization.scene.mlab.test_points3d()
        #self.mayavi_widget.visualization.scene.mlab.clf()
        X = np.zeros([3]); Y = np.zeros([3]); Z = np.mgrid[0 : 10 : 3j]
        self.widget.visualization.scene.mlab.plot3d(X, Y, Z, color = (0, 0, 0), line_width =3, figure=name)
        self.widget.visualization.scene.mlab.plot3d(X, Y+10, Z, color = (0, 0, 0), line_width =3)
        self.widget.visualization.scene.mlab.plot3d(X+10, Y, Z, color = (0, 0, 0), line_width =3)
        self.widget.visualization.scene.mlab.plot3d(X+10, Y+10, Z, color = (0, 0, 0), line_width =3)
        
        X = np.zeros([3]); Z = np.zeros([3]); Y = np.mgrid[0 : 10 : 3j]
        self.widget.visualization.scene.mlab.plot3d(X, Y, Z, color = (0, 0, 0), line_width =3)
        self.widget.visualization.scene.mlab.plot3d(X+10, Y, Z, color = (0, 0, 0), line_width =3)
        self.widget.visualization.scene.mlab.plot3d(X+10, Y, Z+10, color = (0, 0, 0), line_width =3)
        self.widget.visualization.scene.mlab.plot3d(X, Y, Z+10, color = (0, 0, 0), line_width =3)
        Z = np.zeros([3]); Y = np.zeros([3]); X = np.mgrid[0 : 10 : 3j]
        self.widget.visualization.scene.mlab.plot3d(X, Y, Z, color = (0, 0, 0), line_width =3)
        self.widget.visualization.scene.mlab.plot3d(X, Y+10, Z, color = (0, 0, 0), line_width =3)
        self.widget.visualization.scene.mlab.plot3d(X, Y, Z+10, color = (0, 0, 0), line_width =3)
        self.widget.visualization.scene.mlab.plot3d(X, Y+10, Z+10, color = (0, 0, 0), line_width =3)
        # self.mayavi_widget.visualization.scene.parallel_projection = True
        # self.mayavi_widget.visualization.scene.isometric_view = True
        # self.mayavi_widget.visualization.scene.camera.azimuth(5)
        # self.mayavi_widget.visualization.scene.mlab.move(1, -1, -1)

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

    def draw_box2container(self, name, box, color_box):
        # self.model1.run_step()
        # box = self.model1.box[self.i]
        
        x, y, z = box.lx, box.ly, box.lz
        dx,dy,dz = box.x, box.y, box.z
        #print(x, y, z, dx, dy, dz)
        # color_box = tuple(np.random.rand(1, 3)[0])
        #x = 10; y = 10; z = 10; dx = 10; dy = 10; dz = 10
        Y, Z= np.mgrid[y : y+dy : 3j, z : z+dz : 3j]
        X = np.ones([3, 3])*x
        self.mayavi_widget.visualization.scene.mlab.mesh(X, Y, Z, color=color_box, figure = name)
        self.mayavi_widget.visualization.scene.mlab.mesh(X+dx, Y, Z, color=color_box)

        Y, X= np.mgrid[y : y+dy : 3j, x : x+dx : 3j]
        Z = np.ones([3, 3])*z
        self.mayavi_widget.visualization.scene.mlab.mesh(X, Y, Z, color=color_box)
        self.mayavi_widget.visualization.scene.mlab.mesh(X, Y, Z+dz, color=color_box)

        Z, X= np.mgrid[z : z+dz : 3j, x : x+dx : 3j]
        Y = np.ones([3, 3])*y
        self.mayavi_widget.visualization.scene.mlab.mesh(X, Y, Z, color=color_box)
        self.mayavi_widget.visualization.scene.mlab.mesh(X, Y+dy, Z, color=color_box)

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

    # def run_model_full(self):
    #     PCT_policy.eval()
    #     envs.type_gen_box = "Manual"
    #     envs.next_box_cf = (5, 5, 5)
    #     obs = envs.reset()
    #     obs = torch.FloatTensor(obs).to(device).unsqueeze(dim=0)
    #     all_nodes, leaf_nodes = tools.get_leaf_nodes_with_factor(obs, args_num_processes, args_internal_node_holder, args_leaf_node_holder)

    #     batchX = torch.arange(args_num_processes)
    #     step_counter = 0
    #     episode_ratio = []
    #     episode_length = []
    #     all_episodes = []

    #     args_evaluation_episodes = 1 #Number of episodes evaluated
    #     eval_freq = args_evaluation_episodes
    #     args_normFactor = 1.0 / np.max(args_container_size)
    #     factor = args_normFactor
    #     frames = []
    #     while step_counter < eval_freq:
    #         with torch.no_grad():
    #             selectedlogProb, selectedIdx, policy_dist_entropy, value = PCT_policy(all_nodes, True, normFactor = factor)
    #         selected_leaf_node = leaf_nodes[batchX, selectedIdx.squeeze()]
    #         items = envs.packed
    #         #my_next_box = (2, 2, 2)
            
    #         obs, reward, done, infos = envs.step(selected_leaf_node.cpu().numpy()[0][0:6], (5, 5, 5))#, my_next_box)
    #         #frames.append(envs.render(mode='rgb_array'))
    #         print(len(envs.space.boxes[:-1]))
    #         self.box = envs.space.boxes[:-1]
    #         # for box in envs.space.boxes[:-1]:
    #         #     x, y, z = box.lx, box.ly, box.lz
    #         #     dx,dy,dz = box.x, box.y, box.z
    #         #     print(x, y, z, dx, dy, dz)

    #         if done:
    #             print('Episode {} ends.'.format(step_counter))
    #             if 'ratio' in infos.keys():
    #                 episode_ratio.append(infos['ratio'])
    #             if 'counter' in infos.keys():
    #                 episode_length.append(infos['counter'])

    #             print('Mean ratio: {}, length: {}'.format(np.mean(episode_ratio), np.mean(episode_length)))
    #             print('Episode ratio: {}, length: {}'.format(infos['ratio'], infos['counter']))
    #             all_episodes.append(items)
    #             step_counter += 1
    #             obs = envs.reset()

    #         obs = torch.FloatTensor(obs).to(device).unsqueeze(dim=0)
    #         all_nodes, leaf_nodes = tools.get_leaf_nodes_with_factor(obs, args_num_processes, args_internal_node_holder, args_leaf_node_holder)               

def main():
    app = QtWidgets.QApplication(sys.argv)
    ex = window()
    ex.showMaximized()
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

