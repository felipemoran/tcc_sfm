import os
import cv2
import glob
import time
import copy
import random
import argparse
import itertools
import collections
import numpy as np
import open3d as o3d
import networkx as nt
import matplotlib.pyplot as plt

from math import pi

camera_matrix = np.array([
    [200, 0, 400],
    [0, 200, 400],
    [0, 0, 1],
], dtype=np.float_)

points_3d = np.array(list(itertools.product([9, 11], [4, 6], [-1, 1])), dtype=np.float_)

r0_1 = cv2.Rodrigues(np.array([-pi/2, 0., 0.]))[0]
r1_3 = cv2.Rodrigues(np.array([0., pi/2, 0]))[0]
r1_4 = cv2.Rodrigues(np.array([0., 0, pi/2]))[0]

ts = np.array([
    [-10, 0, 0],
    [-5, 0, 15],
], dtype=np.float_)

R_vecs = np.array([
    cv2.Rodrigues(r0_1)[0],
    cv2.Rodrigues(np.matmul(r1_3, r1_4))[0],
])

for R_vec, t in zip(R_vecs, ts):
    projected_points = cv2.projectPoints(points_3d, R_vec, t, camera_matrix, None)[0]
    for p3d, p2d in zip(points_3d, projected_points):
        print(p3d, p2d)
    print()

projected_points_1 = cv2.projectPoints(points_3d, R_vecs[0], ts[0], camera_matrix, None)[0]
projected_points_2 = projected_points_1.copy()

projected_points_2[0] = projected_points_1[4]
projected_points_2[1] = projected_points_1[5]
projected_points_2[2] = projected_points_1[0]
projected_points_2[3] = projected_points_1[1]
projected_points_2[4] = projected_points_1[6]
projected_points_2[5] = projected_points_1[7]
projected_points_2[6] = projected_points_1[2]
projected_points_2[7] = projected_points_1[3]

stat, r, t = cv2.solvePnP(points_3d, projected_points_2, camera_matrix, None, flags=cv2.SOLVEPNP_ITERATIVE)
print(r)
print(t)

a = 1
# points_3d = np.array([
#     [-3, 5, 1],
#     [-2, 5, 1],
#     [-1, 5, 1],
#     [0, 5, 1],
#     [1, 5, 1],
#     [2, 5, 1],
#     [3, 5, 1],
#     [-3, 5, 0],
#     [-2, 5, -1],
#     [-1, 5, -2],
#     [0, 5, 0],
#     [1, 5, -1],
#     [2, 5, -2],
#     [3, 5, -3],
# ], dtype=np.float_)
# points_2d = np.array([
#     [-300+400, -100+600],
#     [-200+400, -100+600],
#     [-100+400, -100+600],
#     [+000+400, -100+600],
#     [+100+400, -100+600],
#     [+200+400, -100+600],
#     [+300+400, -100+600],
#
#     [-300+400, +000+600],
#     [-200+400, +100+600],
#     [-100+400, +200+600],
#     [+000+400, +000+600],
#     [+100+400, +100+600],
#     [+200+400, +200+600],
#     [+300+400, +300+600],
# ], dtype=np.float_)

a = 1
