#!/usr/bin/env python
# coding: utf-8

# ## (Stereo) Visual Odometry

# In[ ]:
class Dataset_Handler():
    def __init__(self, lidar=False, progress_bar=True):
        import pandas as pd
        import os
        import cv2

        self.left_image_files = os.listdir('./data/image_02/data')
        self.right_image_files = os.listdir('./data/image_03/data')
        self.left_image_files.sort()
        self.right_image_files.sort()

        self.num_frames = len(self.left_image_files)

        self.reset_frames()
        # Store original frame to memory for testing functions
        self.first_image_left = cv2.imread('./data/image_02/data/'
                                           + self.left_image_files[0], 0)

        self.first_image_right = cv2.imread('./data/image_03/data/'
                                            + self.right_image_files[0], 0)
        self.second_image_left = cv2.imread('./data/image_02/data/'
                                            + self.left_image_files[1], 0)
        self.imheight = self.first_image_left.shape[0]
        self.imwidth = self.first_image_left.shape[1]

    def reset_frames(self):
        # Resets all generators to the first frame of the sequence
        self.images_left = (cv2.imread('./data/image_02/data/' + name_left, 0)
                            for name_left in self.left_image_files)
        self.images_right = (cv2.imread('./data/image_03/data/' + name_right, 0)
                             for name_right in self.right_image_files)

    def __len__(self):
        return min(len(self.left_image_files), len(self.right_image_files))

    def __getitem__(self, idx):
        return (cv2.imread('./data/image_02/data/'
                           + self.left_image_files[idx], 0),
                cv2.imread('./data/image_03/data/' + self.right_image_files[idx], 0))

handler = Dataset_Handler()

import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import cv2



# ## Extended Kalman Filter
# 
# Now please try to use EKF to fuse the visual odometry from cameras with gps.
# 
# For simplicity, set the state vector X = \begin{bmatrix}
#  b \\
#  x \\
#  y
# \end{bmatrix}
# ### Motion Model
# \begin{align}
# \mathbf{x}_{k} &= 
# \begin{bmatrix}
# 1 &0 & 0 \\
# 0 &1 & 0\\
# 0 &0 & 1
# \end{bmatrix}
# \mathbf{x}_{k-1} +
# \begin{bmatrix}
# 1 &0 \\
# 0 &sin(b)\\
# 0 &cos(b)
# \end{bmatrix}
# \left(
# \begin{bmatrix}
# \theta \\
# d
# \end{bmatrix}
# + \mathbf{w}_k
# \right)
# \, , \, \, \, \, \, \mathbf{w}_k = \mathcal{N}\left(\mathbf{0}, \mathbf{Q}\right)
# \end{align}
# 
# - $\mathbf{x}_k = \left[ b \, x \, y  \right]^T$ is the current bearing and 2d position of the vehicle
# - $\theta $ is the change in bearing between frame k-1 and k, data is stored in "relative_angle"
# - $d$ is the distance traveled between frame k-1 and k, data is stored in "distance"
# 
# The process noise $\mathbf{w}_k$ has a (zero mean) normal distribution with a constant covariance $\mathbf{Q}$.
# 
# 
# ### Measurement Model
# 
# The measurement model from gps $\mathbf{y}_k = \left[x \, y \right]^T$.
# 
# \begin{align}
# \mathbf{y}_k =
# \begin{bmatrix}
# 0 & 1 & 0 \\
# 0 & 0 & 1
# \end{bmatrix}
# x_k
# + \mathbf{n}_k
# \, , \, \, \, \, \, \mathbf{n}_k = \mathcal{N}\left(\mathbf{0}, \mathbf{R}\right)
# \end{align}
# 
# 
# The gps measurement noise $\mathbf{n}_k$ has a (zero mean) normal distribution with a constant covariance $\mathbf{R}$.
# In[ ]:

imu = pd.read_csv("./data/WiFi_IMU/pdr.csv")
imu = imu[['tmsp','x','y']].to_numpy() #time, odometery frame x, odometery frame y
wifi = pd.read_csv("./data/WiFi_IMU/wifi.csv")
wifi = wifi[['time','x','y','uncertainty_normalized']].to_numpy() #time, map frame x, map frame y, uncertainty

# imu = imu[::len(imu)//len(wifi)][:len(wifi)]

pos_var = 1e-1
imu_var = 1e-1
Q_km = np.diag([pos_var, pos_var, imu_var, imu_var])
cov_y = np.diag([imu_var, imu_var])

x_init = np.array(wifi[0, 1:3].tolist() + imu[0, 1:].tolist())
x_hist = np.zeros([len(imu), 4])
x_hist[0] = x_init

x_naive = np.zeros([len(imu), 4])
x_naive[0] = x_init

x_wifi  = np.zeros([len(wifi), 4])
x_wifi[0] = x_init

x_imu  = np.zeros([len(imu), 4])
x_imu[0] = x_init

P_hist_imu = np.zeros([len(imu), 4, 4])  # state covariance matrices
P_hist_imu[0] = np.eye(4)

P_hist = np.zeros([len(wifi), 4, 4])  # state covariance matrices
P_hist[0] = np.eye(4)
curr_wifi = wifi[0]
wifi_idx = 0

for i in range(1, len(imu)):
    delta_t = (imu[i, 0] - imu[i - 1, 0]) / 1e5

    curr_imu = imu[i]
    a_k = np.array([curr_imu[1], curr_imu[2]])

    # Previous state vector [x, y, x*, y*]
    x_k1 = x_hist[i - 1]
    #x_k1 = x_wifi[wifi_idx - 1]

    F_k1 = np.mat([[1, 0, delta_t, 0],
                   [0, 1, 0, delta_t],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]]
                  )
    L_k1 = np.mat([[0.5 * delta_t ** 2, 0],
                   [0, 0.5 * delta_t ** 2],
                   [delta_t,0],
                   [0, delta_t]
                  ])



    #P_k1 = P_hist[wifi_idx - 1]
    P_k1 = P_hist_imu[i - 1]

    # Prediction step using previous state and motion input
    x_k = F_k1 @ x_k1 + L_k1 @ a_k

    # Predicted covariance estimate
    P_est = F_k1 @ P_k1 @ F_k1.T + Q_km
    x_naive[i] = x_k
    if abs(curr_imu[0] - curr_wifi[0]) > 1.3e6:
        x_hist[i] = x_k
        P_hist_imu[i] = P_est
        continue

    print(i)
    H = np.mat([[1, 0, 0, 0],
                [0, 1, 0, 0]])

    y_k = np.mat([[curr_wifi[1]],
                  [curr_wifi[2]]])
    H_k = H
    M_k = np.eye(2)
    cov_y = np.eye(2) * curr_wifi[3]
    K = P_est @ H_k.T @ np.linalg.inv(H_k @ P_est @ H_k.T + M_k.dot(cov_y).dot(M_k.T))


    x_pred = x_k + (K @ (y_k - H @ x_k.T)).T

    print((K @ H_k).shape, (1 - K @ H_k).shape, P_est.shape)
    P_check = (P_est - K @ H_k) @ P_est

    print("f_k1", F_k1)
    print("x_k1: ", x_k1)
    print("P_est: ", P_est)
    print("h_k: ", H_k)
    print("K: ", K)
    print("y_k: ", y_k)
    print("x_k: ", x_k)
    print("x_pred: ", x_pred)
    print("P_check: ", P_check)

    x_hist[i] = x_pred
    x_wifi[wifi_idx] = x_pred
    P_hist[wifi_idx] = P_check
    wifi_idx += 1
    curr_wifi = wifi[wifi_idx]



# In[ ]:


### your implementation:


# Plot your result on the floor plan:

# In[ ]:


def meter2pixel(x,y,fig_resolution=72,fig_scale=100):
    pix_x=x/0.0254/fig_scale*fig_resolution
    pix_y=-y/0.0254/fig_scale*fig_resolution
    return pix_x,pix_y

floorplan = plt.imread("./data/WiFi_IMU/F1.png")
gt = np.load("./data/WiFi_IMU/gt_9.npy")
gtx_pixel,gty_pixel= meter2pixel(gt[:,1],gt[:,2])

plt.imshow(floorplan)
plt.plot(gtx_pixel,gty_pixel,label="GT")
plt.legend()

plt.show()

print(x_hist.shape)

predx_pixel,predy_pixel= meter2pixel(x_hist[:,0],x_hist[:,1])
naivex_pixel, naivey_pixel = meter2pixel(x_naive[:,0], x_naive[:,1])
wifix, wifiy = meter2pixel(x_wifi[:,0], x_wifi[:,1])
originalx, originaly = meter2pixel(wifi[:,1], wifi[:,2])

plt.imshow(floorplan)
# plt.scatter(naivex_pixel,naivey_pixel,label="GT", color='r')
plt.scatter(wifix, wifiy, label="WiFi", color='green')
plt.plot(gtx_pixel,gty_pixel,label="GT", color='b')
plt.scatter(wifix, wifiy, label="WiFi", color='green')
plt.scatter(originalx, originaly, label="Original", color='yellow')
# plt.scatter(predx_pixel[:-2], predy_pixel[:-2], label="WiFi", color='purple')
plt.legend()
plt.show()