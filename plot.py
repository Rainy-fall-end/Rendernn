import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import math
def plot2():
    X = []
    Y = []
    Z = []
    Z1 = []
    Z2 = []
    Z3 = []
    p1_ls = []
    p2_ls = []
    with open("datas/sph_6.json") as f:
    #with open("datas/all_dir_sph_range_4.json") as f:
        datas = json.load(f)
        for data in datas:
            if(float(data["rgb"][0]-0)**2+float(data["rgb"][1]-0)**2+float(data["rgb"][2]-1)**2<1e-3):
                continue
            # if data["hit"] == 0:
            # #     continue
            # p1 = float(data["point_sph"][0])
            # p2 = float(data["point_sph"][1])
            p1 = float(data["dir_sph"][0])
            p2 = float(data["dir_sph"][1])
            X.append(100*math.sin(p1)*math.cos(p2))
            Y.append(100*math.sin(p1)*math.sin(p2))
            Z.append(100*math.cos(p1))
            Z1.append(float(data["rgb"][0]))
            Z2.append(float(data["rgb"][1]))
            Z3.append(float(data["rgb"][2]))
            p1_ls.append(p1)
            p2_ls.append(p2)
    # for i in range(1000):
    #     print(X[i]**2+Y[i]**2+Z[i]**2)
    size = 1000
    print(np.min(p1_ls))
    print(np.max(p1_ls))
    print(np.min(p2_ls))
    print(np.max(p2_ls))
    print(np.max(X))
    print(np.min(X))
    print(np.max(Y))
    print(np.min(Y))
    print(np.max(Z))
    print(np.min(Z))
    X = np.array(X[0:size])
    Y = np.array(Y[0:size])
    Z = np.array(Z[0:size])
    Z1 = np.array(Z1[0:size])
    Z2 = np.array(Z2[0:size])
    Z3 = np.array(Z3[0:size])  
    normalized_Z1 = (Z1 - Z1.min()) / (Z1.max() - Z1.min())
    normalized_Z2 = (Z2 - Z2.min()) / (Z2.max() - Z2.min())
    normalized_Z3 = (Z3 - Z3.min()) / (Z3.max() - Z3.min())

    cmap = plt.cm.plasma
    fig = plt.figure(figsize=(15, 5))
    elev = 35
    azim = 25
    colors = np.vstack((Z1,Z2,Z3)).T
    # the first plot
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(X, Y, Z, c=colors/255.0, cmap=cmap, marker='o')
    ax1.set_title('Scatter Plot 1')
    ax1.set_xlabel('X axis')
    ax1.set_ylabel('Y axis')
    ax1.set_zlabel('Z axis')
    ax1.view_init(elev=elev, azim=azim)

    # the second subplot
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(X, Y, Z, c=normalized_Z2, cmap=cmap, marker='^')
    ax2.set_title('Scatter Plot 2')
    ax2.set_xlabel('X axis')
    ax2.set_ylabel('Y axis')
    ax2.set_zlabel('Z axis')
    ax2.view_init(elev=elev, azim=azim)

    # the third subplot
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(X, Y, Z, c=normalized_Z3, cmap=cmap, marker='x')
    ax3.set_title('Scatter Plot 3')
    ax3.set_xlabel('X axis')
    ax3.set_ylabel('Y axis')
    ax3.set_zlabel('Z axis')
    ax3.view_init(elev=elev, azim=azim)

    # show plot
    plt.show()
plot2()