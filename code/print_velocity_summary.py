import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.stats as scst
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--sim', help='simulation name', required=True)
args = parser.parse_args()

pos = np.loadtxt('../data/summary_pos_{}.dat'.format(args.sim))
vel = np.loadtxt('../data/summary_vel_{}.dat'.format(args.sim))
mass = np.loadtxt('../data/summary_mass_{}.dat'.format(args.sim))
print(len(mass))

d = np.sqrt((pos[:,3]-pos[:,0])**2 + (pos[:,4]-pos[:,1])**2 + (pos[:,5]-pos[:,2])**2)
ii = d < 2000
d = d[ii]

vel = vel[ii]
pos = pos[ii]
mass = mass[ii]

ii = mass[:,0]>mass[:,1]
tmp_vel = vel[ii,0:3].copy()
vel[ii,0:3] = vel[ii,3:6].copy()
vel[ii,3:6] = tmp_vel[:,0:3].copy()

tmp_pos = pos[ii,0:3].copy()
pos[ii,0:3] = pos[ii,3:6]
pos[ii,3:6] = tmp_pos[:,0:3]

tmp_mass = mass[ii,0]
mass[ii,0] = mass[ii,1].copy()
mass[ii,1] = tmp_mass[:]

total_mass = mass[:,0]+ mass[:,1]
mass_ratio = mass[:,0]/mass[:,1]


delta_v = np.zeros([len(d),3])
for i in range(3):
    delta_v[:,i] = vel[:,i+3] - vel[:,i]

delta_r = np.zeros([len(d),3])
for i in range(3):
    delta_r[:,i] = pos[:,i+3] - pos[:,i]

cm_vel = np.zeros([len(d),3])
for i in range(3):
    cm_vel[:,i] = (mass[:,0]*vel[:,i] + mass[:,1]*vel[:,i+3])/(total_mass)
    
norm_cm_vel = np.sqrt(np.sum(cm_vel**2, axis=1))
    
cm_vel_hat = cm_vel.copy()
for i in range(3):
    cm_vel_hat[:,i] = cm_vel_hat[:,i]/norm_cm_vel
    
r_hat = delta_r.copy()
norm_delta_r = np.sqrt(np.sum(delta_r*delta_r, axis=1))
for i in range(3):
    r_hat[:,i] = delta_r[:,i]/norm_delta_r


v_hat = delta_v.copy()
norm_delta_v = np.sqrt(np.sum(delta_v*delta_v, axis=1))
for i in range(3):
    v_hat[:,i] = delta_v[:,i]/norm_delta_v

mu_vr = np.zeros(len(v_hat))
for i in range(3):
    mu_vr += v_hat[:,i] * r_hat[:,i]
    
mu_vcmr = np.zeros(len(cm_vel_hat))
for i in range(3):
    mu_vcmr += cm_vel_hat[:,i] * r_hat[:,i]

v_rad = np.zeros([len(delta_v),3])
for i in range(3):
    v_rad[:,i] = delta_v[:,i]*r_hat[:,i]

norm_v_rad = np.zeros(len(d))
for i in range(3):
    norm_v_rad[:] += delta_v[:,i]*r_hat[:,i]

v_rad = np.zeros([len(d),3])
for i in range(3):
    v_rad[:,i] = norm_v_rad[:] * r_hat[:,i]
    
    
v_tan = np.zeros([len(d),3])
for i in range(3):
    v_tan[:,i] = delta_v[:,i] - v_rad[:,i]
norm_v_tan = np.sqrt(np.sum(v_tan*v_tan, axis=1))

# this includes now hubble flow
delta_v_hflow = delta_v.copy()
for i in range(3):
    delta_v_hflow[:,i] = delta_v[:,i] + 0.1*delta_r[:,i]

norm_delta_v_hflow = np.zeros(len(d))
for i in range(3):
    norm_delta_v_hflow[:] += delta_v_hflow[:,i]*r_hat[:,i]
    
ii = (norm_v_rad<0) & (norm_delta_r<1000)


results  = np.array([norm_cm_vel[ii], norm_v_tan[ii], norm_v_rad[ii]])

print(np.shape(results))



# write positions
fileout = '../data/summary_velocities_{}.dat'.format(args.sim)
np.savetxt(fileout, results.T, fmt='%f %f %f')
print(' wrote results data to {}'.format(fileout))

