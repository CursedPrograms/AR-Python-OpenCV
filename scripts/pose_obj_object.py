"""Draw a 3D OBJ model (objects/cube.obj) standing on each ArUco marker, with OpenGL.

    python scripts/pose_obj_object.py                     # webcam
    python scripts/pose_obj_object.py --image photo.jpg
    python scripts/pose_obj_object.py --model objects/cube.obj --marker-length 0.05
    Keys: q / Esc = quit
"""
import os
import sys

import numpy as np
import cv2
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

import ar_common

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

def init_ar(width, height):
    pygame.init()
    pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("AR OBJ model")

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    # Directional light from the upper left, behind the camera (eye space), so the
    # cube's faces get different shades
    glLightfv(GL_LIGHT0, GL_POSITION, (-0.6, 1.0, 0.8, 0.0))
    glLightfv(GL_LIGHT0, GL_AMBIENT, (0.3, 0.3, 0.3, 1))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.8, 0.8, 0.8, 1))
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    # The model is scaled down a lot; keep its normals unit length for lighting
    glEnable(GL_NORMALIZE)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)  # frame rows are not always 4-byte aligned

def set_projection_from_camera(K, width, height, near=0.01, far=100.0):
    """OpenGL projection matching the camera matrix exactly, including the principal
    point (gluPerspective assumes it is at the image centre)."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    P = np.array([
        [2 * fx / width, 0, 1 - 2 * cx / width, 0],
        [0, 2 * fy / height, 2 * cy / height - 1, 0],
        [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
        [0, 0, -1, 0],
    ])
    glMatrixMode(GL_PROJECTION)
    glLoadMatrixd(P.T)  # OpenGL wants column-major
    glViewport(0, 0, width, height)

def set_modelview_from_camera(rvec, tvec):
    """Marker -> camera transform as the OpenGL model-view matrix. OpenCV's camera
    looks down +z with y down; OpenGL's looks down -z with y up, hence the flip.
    (The old code loaded the inverse of this, which is the camera pose seen from
    the marker, so the model did not sit on the marker.)"""
    R, _ = cv2.Rodrigues(rvec)
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = tvec.ravel()
    M = np.diag([1, -1, -1, 1]) @ M
    glMatrixMode(GL_MODELVIEW)
    glLoadMatrixd(M.T)

def draw_background(frame):
    h, w = frame.shape[:2]
    glDisable(GL_DEPTH_TEST)
    glDisable(GL_LIGHTING)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(0, w, 0, h)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    bg_image = cv2.cvtColor(cv2.flip(frame, 0), cv2.COLOR_BGR2RGB)

    glEnable(GL_TEXTURE_2D)
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, bg_image)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    glColor3f(1.0, 1.0, 1.0)
    glBegin(GL_QUADS)
    glTexCoord2f(0.0, 0.0); glVertex2f(0, 0)
    glTexCoord2f(1.0, 0.0); glVertex2f(w, 0)
    glTexCoord2f(1.0, 1.0); glVertex2f(w, h)
    glTexCoord2f(0.0, 1.0); glVertex2f(0, h)
    glEnd()

    glDeleteTextures([texture_id])
    glDisable(GL_TEXTURE_2D)
    glEnable(GL_LIGHTING)
    glEnable(GL_DEPTH_TEST)

def save_screen(path, width, height):
    glReadBuffer(GL_BACK)
    data = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    img = np.frombuffer(data, np.uint8).reshape(height, width, 3)
    cv2.imwrite(path, cv2.cvtColor(cv2.flip(img, 0), cv2.COLOR_RGB2BGR))
    print(f"Saved {path}")

def main():
    p = ar_common.parser("Draw a 3D OBJ model on ArUco markers (OpenGL)")
    ar_common.add_source_args(p)
    ar_common.add_aruco_args(p)
    p.add_argument("--model", default=os.path.join(ar_common.REPO_DIR, "objects", "cube.obj"),
                   help="OBJ file to draw (default objects/cube.obj)")
    args = p.parse_args()

    if args.image:
        still = cv2.imread(args.image)
        if still is None:
            sys.exit(f"Could not read image {args.image}")
        cap = None
    else:
        cap = ar_common.open_capture(args)
        ok, still = cap.read()
        if not ok:
            sys.exit("Could not read from the camera or video")
    height, width = still.shape[:2]

    init_ar(width, height)
    camera_matrix, dist_coeffs = ar_common.camera_intrinsics(width, height)
    detector = ar_common.make_detector(args.dict)
    obj = ObjLoader(args.model, swapyz=True)
    # objects/cube.obj spans -1..1; scale it to the marker and stand it on the marker
    half = args.marker_length / 2

    clock = pygame.time.Clock()
    smooth = {}  # marker id -> smoothed (rvec, tvec), to reduce jitter
    smooth_factor = 0.6
    frame = still  # the first frame (it also gave the window size)
    first = True

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == KEYDOWN and event.key in (K_q, K_ESCAPE)):
                pygame.quit()
                if cap is not None:
                    cap.release()
                return

        if cap is not None and not first:
            ok, frame = cap.read()
            if not ok:
                break
        first = False

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        draw_background(frame)

        corners, ids, _ = detector.detectMarkers(frame)
        if ids is not None:
            set_projection_from_camera(camera_matrix, width, height)
            for c, marker_id in zip(corners, ids.flatten()):
                pose = ar_common.estimate_pose(c, args.marker_length, camera_matrix, dist_coeffs)
                if pose is None:
                    continue
                rvec, tvec = pose
                if marker_id in smooth:
                    sr, st = smooth[marker_id]
                    rvec = smooth_factor * sr + (1 - smooth_factor) * rvec
                    tvec = smooth_factor * st + (1 - smooth_factor) * tvec
                smooth[marker_id] = (rvec, tvec)

                set_modelview_from_camera(rvec, tvec)
                glColor3f(1.0, 0.65, 0.25)
                glPushMatrix()
                glTranslatef(0, 0, half)       # base on the marker, not halfway through it
                glScalef(half, half, half)
                obj.render()
                glPopMatrix()

        if args.save:
            save_screen(args.save, width, height)
            if cap is None or args.no_show:
                pygame.quit()
                return
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    if cap is not None:
        cap.release()

if __name__ == "__main__":
    main()