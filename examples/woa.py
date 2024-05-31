import math
import random
import time

import numpy as np

# for i in range(10):
#     print(i)
#
# # 3d py list
# board_obs_l = [[[0 for _ in range(3)] for _ in range(3)] for _ in range(3)]
# test = np.array(board_obs_l)
# print(test.shape)
#
# # 3d np array
# board_obs = np.zeros((3, 3, 3))
#
# for o in board_obs:
#     print(o.shape)

# Step 1: Create a 1D array with values from 1 to 9
# array_1d = np.arange(1, 10)
# array_2d = np.reshape(array_1d, (3, 3))
# print(array_2d)
#
# boards = []
# for _ in range(8):
#     boards.append(array_2d)
#
# boards = np.array(boards)
# print(boards.shape)
#
# # H x W
#
# c = 4
# n, w, h = boards.shape[0], boards.shape[1], boards.shape[2]
#
# rotations = np.split(boards, n // c)
# cols = []
# for r in rotations:
#     cols.append(np.vstack(r))
# result = np.hstack(cols)
# print(result)
#
#
# c = 4
# n, w, h = boards.shape[0], boards.shape[1], boards.shape[2]
#
# # Split the boards array into chunks along the first axis
# rotations = np.split(boards, n // c, axis=0)
#
# # Stack the chunks vertically
# cols = np.vstack(rotations)
#
# # Reshape the stacked columns to merge them horizontally
# result = cols.reshape(n // c, w * c, h).transpose(1, 0, 2).reshape(w * c, h)
#
# print(result)

greek_letters = [
    "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta",
    "Iota", "Kappa", "Lambda", "Mu", "Nu", "Xi", "Omicron", "Pi", "Rho",
    "Sigma", "Tau", "Upsilon", "Phi", "Chi", "Psi", "Omega"
]
greek_letters = [letter.lower() for letter in greek_letters]

file_name = [
    "train_cnn", "train_cnn_grouped", "train_cnn_grouped_rgb",
]

seed = 1

print(f"{random.choice(file_name)}/{random.choice(greek_letters)}_{random.choice(greek_letters)}_{seed}_{int(time.time())}")