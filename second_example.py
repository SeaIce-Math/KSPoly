#from sklearn import *

import kspoly as ks
import numpy as np 

from matplotlib import pyplot as plt
import matplotlib.cm as cm




sc = ks.scene(sigma = 0.4, mesh_size=[60,60], xlim = [-8,8], ylim = [-10,10])

poly1 = np.array([[0,2,1],[0,0,2]])
poly2 = np.array([[5,6,6,4],[1,1,3,3]])

poly_land1 = np.array([[-4,-4,-3,-3], [1,-1,-1,1]])
poly_land2 = np.array([[-6,-6,-4,-4], [2,-2,-2,2]])

v1 = np.array([-1,1])
v2 = np.array([1,0])

st1  = np.array([[1,2],[2,1]])
st2  = np.array([[1,-1],[0,1]])

#video = ks.video_generator('mass_video.mp4')

sc.set_land([poly_land1, poly_land2])

for i in range(3):
    poly1 = poly1 + 0.5*i*v1.reshape(-1,1)
    poly2 = poly2 + 0.5*i*v2.reshape(-1,1)
    frame = ks.frame_data([poly1, poly2], velocities=[v1, v2])
    frame.set_field([st1, st2], name='stress')
    frame.set_field([2, 3], name='mass')    
    sm = sc.smoothit(frame)
    sm_vel = sm['velocity']
    sm_mass = sm['mass']
    sm_stress = sm['stress']
    #video.add_frame(sm_mass)

    # Make plots of smooth fields 
    mass = np.squeeze(sm_mass)
    v_x = np.squeeze(sm_vel[:,:,0])
    v_y = np.squeeze(sm_vel[:,:,1])
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 6), sharey=True, sharex =True, dpi=100)
    axs[0].set_title("smooth mass density", fontsize=18)
    h0 = axs[0].pcolormesh(sc.frame_xx, sc.frame_yy, mass, cmap=cm.Blues , vmin=0.0, vmax=1.0, alpha=1.0)
    cbar = plt.colorbar(h0,ax=axs[0], ticks=[0, 1])
    axs[1].set_title("smooth x component of velocity", fontsize=18)
    h1 = axs[1].pcolormesh(sc.frame_xx, sc.frame_yy, v_x, cmap=cm.RdBu , vmin=-2.0, vmax=2.0, alpha=1.0)
    cbar = plt.colorbar(h1,ax=axs[1], ticks=[-2, 2])
    axs[2].set_title("smooth y component of velocity", fontsize=18)
    h2 = axs[2].pcolormesh(sc.frame_xx, sc.frame_yy, v_y, cmap=cm.RdBu , vmin=-2.0, vmax=2.0, alpha=1.0)
    cbar = plt.colorbar(h2,ax=axs[2], ticks=[-2, 2])
    plt.show()

#video.done()