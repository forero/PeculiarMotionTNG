import numpy as np
import illustris_python.groupcat as gc

subhalos = gc.loadSubhalos('/global/cscratch1/sd/forero/TNG/TNG300-1/', 99, fields=['SubhaloGrNr', 'SubhaloParent', 'SubhaloLen', 'SubhaloCM'])

pair_filename = "../data/summary_ids_TNG300-1.dat"
pair_halo_ids = np.int_(np.loadtxt(pair_filename))

pair_subhalo_ids = pair_halo_ids.copy()
subhaloid = np.arange(subhalos['count'], dtype=int)


for i in range(len(pair_halo_ids)):
    print(i, len(pair_halo_ids))
    for j in [0,1]:
        ii = (subhalos['SubhaloGrNr']==pair_halo_ids[i,j])
        pair_subhalo_ids[i,j] = subhaloid[ii][np.argsort(subhalos['SubhaloLen'][ii])[-1]] 

fileout = pair_filename.replace('_ids_', '_subhalo_ids_')
np.savetxt(fileout, np.int_(pair_subhalo_ids), fmt='%d %d')
print(' wrote subhalo ID data to {}'.format(fileout))

fileout = pair_filename.replace('_ids_', '_subhalo_pos_')
np.savetxt(fileout, subhalos['SubhaloCM'][np.int_(pair_subhalo_ids[:,0])], fmt='%f %f %f')
print(' wrote subhalo pos data to {}'.format(fileout))

