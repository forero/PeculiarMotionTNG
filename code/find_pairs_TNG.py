import illustris_python.groupcat as gc
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--sim', help='simulation name', required=True)
args = parser.parse_args()

# Simulation setup
sim_name = args.sim
main_path = "/global/cscratch1/sd/forero/TNG/"
base_path = os.path.join(main_path, sim_name)
halo_fields = ['GroupFirstSub', 'Group_M_Crit200', 'Group_R_Crit200', 'Group_M_Mean200',
               'GroupNsubs', 'GroupPos', 'GroupVel', 'GroupFirstSub', 'GroupMassType']


# Read data
print('Simulation {}'.format(sim_name))
halos = gc.loadHalos(base_path, 99, fields=halo_fields)
halo_stellar_mass = halos['GroupMassType'][:,4]
dm_halo_mass = halos['Group_M_Crit200']
rank_id = np.arange(len(dm_halo_mass), dtype=int)

# Read header
header = gc.loadHeader(base_path,99)
BoxSize = header['BoxSize']
print('Box Size {}'.format(BoxSize))

n_halos = len(halo_stellar_mass)
print('Read {} halos'.format(n_halos))

# Only keep massive galaxies
ii = dm_halo_mass > 50  # in units of 10^10 Msun

# Select the properties of interest
S_pos = halos['GroupPos'][ii]
S_vel = halos['GroupVel'][ii]
S_stellar_mass = halo_stellar_mass[ii]
S_parent_fof = halos['Group_M_Crit200'][ii]
S_rank_id = rank_id[ii]


n_S = len(S_stellar_mass)
print('Kept {} halos'.format(n_S))


#pad boxes around the S3 positions to mimic periodic boundary conditions
S_pad_pos = S_pos.copy()
S_pad_vel = S_vel.copy()
S_pad_stellar_mass = S_stellar_mass.copy()
S_pad_fof = S_parent_fof.copy()
S_pad_id = S_rank_id.copy()
for i in (0,1,-1):
    for j in (0,1,-1):
        for k in (0,1,-1):
            new_pos = S_pos.copy()
            if(i):
                new_pos[:,0] = new_pos[:,0] + i*BoxSize
            if(j):
                new_pos[:,1] = new_pos[:,1] + j*BoxSize
            if(k):
                new_pos[:,2] = new_pos[:,2] + k*BoxSize
                
            if((i!=0) | (j!=0) | (k!=0)):
                S_pad_pos = np.append(S_pad_pos, new_pos, axis=0)
                S_pad_vel = np.append(S_pad_vel, S_vel, axis=0)
                S_pad_stellar_mass = np.append(S_pad_stellar_mass, S_stellar_mass)
                S_pad_id = np.append(S_pad_id, S_rank_id)
                S_pad_fof = np.append(S_pad_fof, S_parent_fof)


print('Startetd Nearest Neighbors')
nbrs_S = NearestNeighbors(n_neighbors=20, algorithm='ball_tree').fit(S_pad_pos)
dist_S, ind_S = nbrs_S.kneighbors(S_pad_pos)
print('Finished Nearest Neighbors')


print('Started finding pairs')
neighbor_index = ind_S[:,1]
neighbor_list = ind_S[:,2:]
#print(np.shape(neighbor_list))

n_pairs = 0
vcm = []
halo_A_id = np.empty((0), dtype=int)
halo_B_id = np.empty((0), dtype=int)
pos_A = []
pos_B = []
vel_A = []
vel_B = []
mass_A = []
mass_B = []
for i in range(n_S):
    l = neighbor_index[neighbor_index[i]]% n_S
    j = neighbor_index[i] % n_S
    
    other_j = neighbor_list[i,:] % n_S
    other_l = neighbor_list[neighbor_index[i],:] % n_S
    
    if((i==l) & (not (j in halo_A_id)) & (not (i in halo_B_id))): # first check to find mutual neighbors
        if((dist_S[i,1] < 1000.0)): #check on the distance between the two galaxies
            halo_mass_i = S_pad_fof[i]
            halo_mass_j = S_pad_fof[j]
            if (halo_mass_i < 5000) and (halo_mass_j < 5000): # check on the stellar mass of the two halos

                mass_limit = min([halo_mass_i, halo_mass_j])
                
                pair_d = dist_S[i,1] # This is the current pair distance
                dist_limit = pair_d * 3.0 # exclusion radius for massive structures

                massive_close_to_i = any((dist_S[i,2:]<dist_limit) & (S_pad_stellar_mass[other_j] >= mass_limit))
                massive_close_to_j = any((dist_S[j,2:]<dist_limit) & (S_pad_stellar_mass[other_l] >= mass_limit))
                if((not massive_close_to_i) & (not massive_close_to_j)): # check on massive structures inside exclusion radius
                    n_pairs = n_pairs+ 1
                    halo_A_id = np.append(halo_A_id, int(S_pad_id[i]))
                    halo_B_id = np.append(halo_B_id, int(S_pad_id[j]))
                    vel_i = S_pad_vel[i,:]
                    vel_j = S_pad_vel[j,:]
                    pos_A.append(S_pad_pos[i,:])
                    pos_B.append(S_pad_pos[j,:])
                    vel_A.append(vel_i)
                    vel_B.append(vel_j)
                    mass_A.append(halo_mass_i)
                    mass_B.append(halo_mass_j)

                    # center of mass velocity
                    m_tot = halo_mass_i + halo_mass_j
                    v = vel_i*halo_mass_i/m_tot + vel_j*halo_mass_j/m_tot
                    v = np.sqrt(np.sum(v**2))
                    vcm.append(v)


print('Found {} siolated pairs pairs'.format(n_pairs))

# write center of mass velocity
#vcm = np.array(vcm)
#fileout = '../data/summary_vcm_{}.dat'.format(sim_name)
#np.savetxt(fileout, vcm.T)
#print(' wrote velocity data to {}'.format(fileout))

# write IDS
pairid = np.array([halo_A_id, halo_B_id])
fileout = '../data/summary_ids_{}.dat'.format(sim_name)
np.savetxt(fileout, np.int_(pairid.T), fmt='%d %d')
print(' wrote ID data to {}'.format(fileout))

# write positions
pos = np.concatenate([pos_A,pos_B], axis=1)
fileout = '../data/summary_pos_{}.dat'.format(sim_name)
np.savetxt(fileout, pos, fmt='%f %f %f %f %f %f')
print(' wrote pos data to {}'.format(fileout))

# write velocities
vel = np.concatenate([vel_A,vel_B], axis=1)
fileout = '../data/summary_vel_{}.dat'.format(sim_name)
np.savetxt(fileout, vel, fmt='%f %f %f %f %f %f')
print(' wrote vel data to {}'.format(fileout))

# write masses
mass = np.array([mass_A,mass_B])
fileout = '../data/summary_mass_{}.dat'.format(sim_name)
np.savetxt(fileout, mass.T, fmt='%f %f')
print(' wrote mass data to {}'.format(fileout))
