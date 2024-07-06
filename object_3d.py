import pygame as pg
from matrix_functions import *
from numba import njit
import cv2
import time

@njit(fastmath=True)
def any_func(arr, a, b):
    return np.any((arr == a) | (arr == b))

class Box_test:
    def __init__(self, x, y, z, dx, dy, dz, index):
        self.x = dx
        self.y = dy
        self.z = dz
        self.lx = x
        self.ly = y
        self.lz = z
        self.index = index

class Object3D:
    def __init__(self, app, box):
        self.app = app
        self.render = app
        x, y, z = box.lx, box.ly, box.lz
        dx,dy,dz = box.x, box.y, box.z
        i = 1
        self.vertices = np.array([(x+dx, y, z, i), (x+dx, y, z+dz, i),(x+dx, y+dy, z+dz, i), (x+dx, y+dy, z, i),
                           (x, y, z, i), (x, y, z+dz, i),(x, y+dy, z+dz, i), (x, y+dy, z, i)])
        self.faces = np.array([(0, 1, 2, 3), (4, 5, 6, 7), (0, 4, 5, 1), (2, 3, 7, 6), (1, 2, 6, 5), (0, 3, 7, 4)])
        self.center_point = np.array([(x+dx/2, y+dy/2, z+dz/2, 1)])#, dtype=int)
        self.color_faces, self.color_vertices = None, [None, None]
        self.draw_faces, self.draw_vertices = None, [None, None]
        self.thinkness_vertices = [None, None]
        

        # khung = 0
        
        # self.vertices = np.array([(0, 0, 0, 1), (0, khung, 0, 1), (khung, khung, 0, 1), (khung, 0, 0, 1),
        #                         (0, 0, khung, 1), (0, khung, khung, 1), (khung, khung, khung, 1), (khung, 0, khung, 1)])
        # self.faces = np.array([(0, 1, 2, 3), (4, 5, 6, 7), (0, 4, 5, 1), (2, 3, 7, 6), (1, 2, 6, 5), (0, 3, 7, 4)])
        # box = Box_test(0, 0, 0, 2, 2, 2, index=1)
        # self.convert_infor(box)
        # box = Box_test(0, 2, 0, 2, 2, 2, index=1)
        # self.convert_infor(box)
        # self.translate([0.0001, 0.0001, 0.0001])

        # self.font = pg.font.SysFont('Arial', 30, bold=True)
        # self.color_faces = [(pg.Color('orange'), face) for face in self.faces]
        # self.movement_flag, self.draw_vertices = True, True
        # self.label = ''

        self.frame = np.zeros((450,800,3),np.uint8)

    def get_distance_camera(self):
        x = self.center_point - self.app.camera.position
        self.distance_center2camera = np.linalg.norm(x)

    def sort_vertices_n_faces(self):
        self.distance_vertices2camera = []
        for point in self.vertices:
            x = point - self.app.camera.position
            self.distance_vertices2camera.append(np.linalg.norm(x))

        # self.vertices = [x for _,x in sorted(zip(self.distance_vertices2camera, self.vertices))]
        ranks = np.argsort(self.distance_vertices2camera)
        self.vertices = self.vertices[ranks]
        self.copy_faces = self.faces
        for i in range(0, 8):
            self.copy_faces[self.faces == i] = ranks[i]
        self.faces = self.copy_faces

    def draw_to_frame(self, frame):
        self.sort_vertices_n_faces()
        vertices = self.vertices @ self.app.camera.camera_matrix()
        vertices = vertices @ self.app.projection.projection_matrix
        vertices /= vertices[:, -1].reshape(-1, 1)
        vertices[(vertices > 2) | (vertices < -2)] = 0
        vertices = vertices @ self.app.projection.to_screen_matrix
        vertices = vertices[:, :2]

        for index, color_face in enumerate(self.color_faces):
            color, face = color_face
            polygon = vertices[face]
            #print(polygon)
            #i =input()
            if not any_func(polygon, self.render.H_WIDTH, self.render.H_HEIGHT):
                
                cv2.fillPoly(self.frame, np.int32([polygon]), [0, 0, 255])
                cv2.polylines(self.frame, np.int32([polygon]), True, [0, 255, 0], 2)
                #cv2.imshow('tung', self.frame)
                #pg.draw.polygon(self.render.screen, color, polygon, 1)

        return frame
    


    def convert_infor(self, box):
        x, y, z = box.lx, box.ly, box.lz
        dx,dy,dz = box.x, box.y, box.z
        i = 1
        new_vertices = np.array([(x+dx, y, z, i), (x+dx, y, z+dz, i),(x+dx, y+dy, z+dz, i), (x+dx, y+dy, z, i),
                           (x, y, z, i), (x, y, z+dz, i),(x, y+dy, z+dz, i), (x, y+dy, z, i)])
        new_faces = np.array([(0, 1, 2, 3), (4, 5, 6, 7), (0, 4, 5, 1), (2, 3, 7, 6), (1, 2, 6, 5), (0, 3, 7, 4)]) + self.vertices.shape[0]
        self.faces = np.concatenate((self.faces, new_faces), axis=0)
        self.vertices = np.concatenate((self.vertices, new_vertices), axis=0)
        self.color_faces = [(pg.Color('orange'), face) for face in self.faces]

    def sort_box(self, boxes, pos_cam):
        for box in boxes:
            x, y, z = box.lx, box.ly, box.lz
            dx,dy,dz = box.x, box.y, box.z
            center_box = np.array([x+dx/2, y+dy/2, z+dz/2], dtype=int)
            


    # def draw(self):
    #     time.sleep(0.03)
    #     self.screen_projection()
    #     self.movement()

    # def movement(self):
    #     if self.movement_flag:
    #         self.rotate_y(-(pg.time.get_ticks() % 0.005))

    def screen_projection(self):
        self.frame = np.zeros((450,800,3),np.uint8)+255
        vertices = self.vertices @ self.app.camera.camera_matrix()
        vertices = vertices @ self.app.projection.projection_matrix
        vertices /= vertices[:, -1].reshape(-1, 1)
        vertices[(vertices > 2) | (vertices < -2)] = 0
        vertices = vertices @ self.app.projection.to_screen_matrix
        vertices = vertices[:, :2]

        for index, color_face in enumerate(self.color_faces):
            color, face = color_face
            polygon = vertices[face]
            #print(polygon)
            #i =input()
            if not any_func(polygon, self.render.H_WIDTH, self.render.H_HEIGHT):
                
                cv2.fillPoly(self.frame, np.int32([polygon]), [0, 0, 255])
                cv2.polylines(self.frame, np.int32([polygon]), True, [0, 255, 0], 2)
                #cv2.imshow('tung', self.frame)
                #pg.draw.polygon(self.render.screen, color, polygon, 1)
                if self.label:
                    text = self.font.render(self.label[index], True, pg.Color('white'))
                    self.render.screen.blit(text, polygon[-1])

        # if self.draw_vertices:
        #     for vertex in vertices:
        #         if not any_func(vertex, self.render.H_WIDTH, self.render.H_HEIGHT):
        #             pg.draw.circle(self.render.screen, pg.Color('white'), vertex, 2)

    def translate(self, pos):
        self.vertices = self.vertices @ translate(pos)
        self.center_point = self.center_point @ translate(pos)

    def scale(self, scale_to):
        self.vertices = self.vertices @ scale(scale_to)
        self.center_point = self.center_point @ scale(scale_to)

    def rotate_x(self, angle):
        self.vertices = self.vertices @ rotate_x(angle)
        self.center_point = self.center_point @ rotate_x(angle)

    def rotate_y(self, angle):
        self.vertices = self.vertices @ rotate_y(angle)
        self.center_point = self.center_point @ rotate_y(angle)

    def rotate_z(self, angle):
        self.vertices = self.vertices @ rotate_z(angle)
        self.center_point = self.center_point @ rotate_z(angle)


class Axes(Object3D):
    def __init__(self, render):
        super().__init__(render)
        self.vertices = np.array([(0, 0, 0, 1), (1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)])
        self.faces = np.array([(0, 1), (0, 2), (0, 3)])
        self.colors = [pg.Color('red'), pg.Color('green'), pg.Color('blue')]
        self.color_faces = [(color, face) for color, face in zip(self.colors, self.faces)]
        self.draw_vertices = False
        self.label = 'XYZ'
