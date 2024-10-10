import numpy as np
import cv2
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

class ObjLoader:
    def __init__(self, filename, swapyz=False):
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []
        self.gl_list = None

        for line in open(filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            
            if values[0] == 'v':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.vertices.append(v)
            elif values[0] == 'vn':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.normals.append(v)
            elif values[0] == 'vt':
                self.texcoords.append(list(map(float, values[1:3])))
            elif values[0] == 'f':
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)
                self.faces.append((face, norms, texcoords))

    def create_gl_list(self):
        if self.gl_list is not None:
            return self.gl_list
        
        self.gl_list = glGenLists(1)
        glNewList(self.gl_list, GL_COMPILE)
        glFrontFace(GL_CCW)
        for face in self.faces:
            vertices, normals, texture_coords = face
            glBegin(GL_POLYGON)
            for i in range(len(vertices)):
                if normals[i] > 0:
                    glNormal3fv(self.normals[normals[i] - 1])
                glVertex3fv(self.vertices[vertices[i] - 1])
            glEnd()
        glEndList()
        return self.gl_list

    def render(self):
        glCallList(self.create_gl_list())

def init_ar():
    pygame.init()
    display = (1280, 720)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glLightfv(GL_LIGHT0, GL_POSITION, (0, 0, -2, 1))
    glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.5, 0.5, 0.5, 1))
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    
    return display

def set_projection_from_camera(intrinsic):
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    
    fx = intrinsic[0,0]
    fy = intrinsic[1,1]
    fovy = 2 * np.arctan(0.5*720 / fy) * 180 / np.pi
    aspect = (1280 * fy) / (720 * fx)

    gluPerspective(fovy, aspect, 0.1, 100.0)
    glViewport(0, 0, 1280, 720)

def set_modelview_from_camera(rvec, tvec):
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    
    rotation = rvec[0][0]
    translation = tvec[0][0]
    
    rmtx = cv2.Rodrigues(rotation)[0]
    
    view_matrix = np.array([[rmtx[0,0], rmtx[0,1], rmtx[0,2], translation[0]],
                           [rmtx[1,0], rmtx[1,1], rmtx[1,2], translation[1]],
                           [rmtx[2,0], rmtx[2,1], rmtx[2,2], translation[2]],
                           [0.0, 0.0, 0.0, 1.0]])
    
    view_matrix = view_matrix * np.array([1, -1, -1, 1])
    
    inverse_matrix = np.linalg.inv(view_matrix)
    glLoadMatrixf(inverse_matrix.T)

def draw_background(frame):
    glDisable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(0, 1280, 0, 720)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    
    # Convert frame to OpenGL texture format
    bg_image = cv2.flip(frame, 0)
    bg_image = cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)
    
    glEnable(GL_TEXTURE_2D)
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1280, 720, 0, GL_RGB, GL_UNSIGNED_BYTE, bg_image)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    
    # Draw textured quad
    glBegin(GL_QUADS)
    glTexCoord2f(0.0, 1.0); glVertex2f(0, 0)
    glTexCoord2f(1.0, 1.0); glVertex2f(1280, 0)
    glTexCoord2f(1.0, 0.0); glVertex2f(1280, 720)
    glTexCoord2f(0.0, 0.0); glVertex2f(0, 720)
    glEnd()
    
    glDeleteTextures([texture_id])
    glDisable(GL_TEXTURE_2D)
    glEnable(GL_DEPTH_TEST)

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    display = init_ar()
    
    camera_matrix = np.array([[933.15867, 0, 657.59],
                             [0, 933.1586, 400.36993],
                             [0, 0, 1]])
    dist_coeffs = np.array([-0.43948, 0.18514, 0, 0])
    
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)
    parameters = cv2.aruco.DetectorParameters_create()
    
    obj = ObjLoader("objects/cube.obj", swapyz=True)
    
    clock = pygame.time.Clock()
    
    # Initialize smoothing variables
    smooth_rvec = None
    smooth_tvec = None
    smooth_factor = 0.8
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                cap.release()
                return
        
        ret, frame = cap.read()
        if not ret:
            break
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        draw_background(frame)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        
        if ids is not None:
            set_projection_from_camera(camera_matrix)
            
            for i in range(len(ids)):
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, camera_matrix, dist_coeffs)
                
                # Apply smoothing
                if smooth_rvec is None:
                    smooth_rvec, smooth_tvec = rvec, tvec
                else:
                    smooth_rvec = smooth_factor * smooth_rvec + (1 - smooth_factor) * rvec
                    smooth_tvec = smooth_factor * smooth_tvec + (1 - smooth_factor) * tvec
                
                set_modelview_from_camera(smooth_rvec, smooth_tvec)
                
                glColor3f(1.0, 1.0, 1.0)  # Set color to white
                glPushMatrix()
                glScalef(0.005, 0.005, 0.005)  # Scale down the cube
                obj.render()
                glPopMatrix()
        
        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()