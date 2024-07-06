import sys
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets

from object_3d import *
from camera import *
from projection import *
import threading
import vtk
print('ok')

from mayavi import mlab

# from ..model import *
# from tools import *
# import givenData
# import gym
# import torch.nn as nn
# from tools import init
# import tools
# from numpy import sqrt
# from attention_model import AttentionModel
# import numpy as np
# import cv2 as cv

# def save_as_video(name_file, size_frame, frames):
#     fourcc = cv.VideoWriter_fourcc(*'XVID')
#     out = cv.VideoWriter(name_file, fourcc, 30.0, size_frame)
#     for frame in frames:
#         out.write(frame)
#         #cv.imshow('frame', frame)
#         if cv.waitKey(1) == ord('q'):
#             break
#     out.release()
#     cv.destroyAllWindows()

# class DRL_GAT(nn.Module):
#     def __init__(self, args_embedding_size,
#                  args_hidden_size,
#                  args_gat_layer_num,
#                  args_internal_node_holder,
#                  args_internal_node_length,
#                  args_leaf_node_holder):
#         super(DRL_GAT, self).__init__()

#         self.actor = AttentionModel(args_embedding_size,
#                                     args_hidden_size,
#                                     n_encode_layers = args_gat_layer_num,
#                                     n_heads = 1,
#                                     internal_node_holder = args_internal_node_holder,
#                                     internal_node_length = args_internal_node_length,
#                                     leaf_node_holder = args_leaf_node_holder,
#                                     )
#         init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), sqrt(2))
#         self.critic = init_(nn.Linear(args_embedding_size, 1))

#     def forward(self, items, deterministic = False, normFactor = 1, evaluate = False):
#         o, p, dist_entropy, hidden, _= self.actor(items, deterministic, normFactor = normFactor, evaluate = evaluate)
#         values = self.critic(hidden)
#         return o, p, dist_entropy,values

#     def evaluate_actions(self, items, actions, normFactor = 1):
#         _, p, dist_entropy, hidden, dist = self.actor(items, evaluate_action = True, normFactor = normFactor)
#         action_log_probs = dist.log_probs(actions)
#         values =  self.critic(hidden)
#         return values, action_log_probs, dist_entropy.mean()

# registration_envs()

# args_id = 'PctContinuous-v0'
# args_setting = 1
# if args_setting == 1:
#     args_internal_node_length = 6
# elif args_setting == 2:
#     args_internal_node_length = 6
# elif args_setting == 3:
#     args_internal_node_length = 7
# args_container_size = givenData.container_size
# args_item_size_set  = givenData.item_size_set
# args_dataset_path = "D:/New folder (5)/20212/do-an-20221/New folder (4)/PCB-dataset/setting123_discrete.pt"
# args_load_dataset = True
# args_internal_node_holder = 80 #Maximum number of internal nodes
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
# #"D:/New folder (5)/20212/do-an-20221/New folder (4)/PCB-model/PCT-model3_tung-2022.11.22-20-40-06_2022.11.22-20-41-14.pt"
# PCT_policy = load_policy(args_model_path, PCT_policy)
# print('Pre-train model loaded!')

class window(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(window, self).__init__(parent)
        self.resize(200, 200)
        # self.setGeometry(100, 100, 600, 400) set kich thuoc window

        self.setWindowTitle("PyQt5")
        self.label = QtWidgets.QLabel(self)
        self.label.setText("Hello World")
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label.setFont(font)
        self.label.move(50, 20)
        self.label.resize(800, 450)

        self.button_show_image = QtWidgets.QPushButton(self)
        self.button_show_image.setText("Button2")
        self.button_show_image.move(50, 5)
        self.button_show_image.clicked.connect(self.test_button)

        self.button_run_model = QtWidgets.QPushButton(self)
        self.button_show_image.setText("run model")
        self.button_show_image.move(150, 5)
        self.button_show_image.clicked.connect(self.test_button)

        self.RES = self.WIDTH, self.HEIGHT = 800, 450
        self.H_WIDTH, self.H_HEIGHT = self.WIDTH // 2, self.HEIGHT // 2
        self.FPS = 60
        self.screen = pg.display.set_mode(self.RES)
        self.clock = pg.time.Clock()
        self.create_objects()

    def keyPressEvent(self, event):
        key = event.key()
        if key == QtCore.Qt.Key.Key_A:
            self.camera.position -= self.camera.right * self.camera.moving_speed
        if key == QtCore.Qt.Key.Key_D:
            self.camera.position += self.camera.right * self.camera.moving_speed
        if key == QtCore.Qt.Key.Key_W:
            self.camera.position += self.camera.forward * self.camera.moving_speed
        if key == QtCore.Qt.Key.Key_S:
            self.camera.position -= self.camera.forward * self.camera.moving_speed
        if key == QtCore.Qt.Key.Key_Q:
            self.camera.position += self.camera.up * self.camera.moving_speed
        if key == QtCore.Qt.Key.Key_E:
            self.camera.position -= self.camera.up * self.camera.moving_speed

        if key == QtCore.Qt.Key.Key_H:
            self.camera.camera_yaw(-self.camera.rotation_speed)
        if key == QtCore.Qt.Key.Key_K:
            self.camera.camera_yaw(self.camera.rotation_speed)
        if key == QtCore.Qt.Key.Key_U:
            self.camera.camera_pitch(-self.camera.rotation_speed)
        if key == QtCore.Qt.Key.Key_J:
            self.camera.camera_pitch(self.camera.rotation_speed)

        
    def create_objects(self):
        self.camera = Camera(self, [0.5, 1, -4])
        self.projection = Projection(self)
        self.object = Object3D(self)
        box = Box_test(0, 0, 0, 2, 2, 2, index=1)
        self.object.convert_infor(box)
        box = Box_test(0, 2, 0, 2, 2, 2, index=1)
        self.object.convert_infor(box)
        box = Box_test(0, 0, 2, 2, 2, 2, index=1)
        self.object.convert_infor(box)
        # self.object.translate([0.2, 0.4, 0.2])
        self.object.rotate_y(math.pi/6)

    def while_image(self):
        while True:
            self.object.draw()
            img = self.object.frame
            h, w, ch = 450, 800, 3
            bytes_per_line = ch * w
            qimage = QtGui.QImage(img.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            self.label.setPixmap(QtGui.QPixmap.fromImage(qimage))

    def test_button(self):
        self.object.draw()
        t1=threading.Thread(target=self.while_image)
        t1.start()

        # pixmap = QtGui.QPixmap(
        #     'D:\\New folder (5)\\20212\\do-an-20221\\New folder (4)\\pygui\Python.png')

        # img = cv2.imread('D:\\New folder (5)\\20212\\do-an-20221\\New folder (4)\\pygui\Python.png')
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # scale_percent = 50 # percent of original size
        # width = int(img.shape[1] * scale_percent / 100)
        # height = int(img.shape[0] * scale_percent / 100)
        # dim = (width, height)
        # img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        # h, w, ch = img.shape
        # bytes_per_line = ch * w
        # qimage = QtGui.QImage(img.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)

        # img = self.object.frame
        # h, w, ch = 450, 800, 1
        # bytes_per_line = ch * w
        # qimage = QtGui.QImage(img.data, w, h, bytes_per_line, QtGui.QImage.Format_Grayscale8)
        # self.label.setPixmap(QtGui.QPixmap.fromImage(qimage))


def main():
    app = QtWidgets.QApplication(sys.argv)
    ex = window()
    ex.showMaximized()
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
