# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 20:29:30 2024

@author: jduij
"""
import numpy as np
import optic_flow_dots

def random_unit_vector():
    V=np.random.sample(3) # create a random 3D translation vecor
    V=V/np.linalg.norm(V) # normalize to unit-length
    V[np.random.sample(V.size)<0.5]*=-1 # randomize sign of each dimension
    return V

def create_optic_flow_files(n=10,
                            trans_speed_minmax: np.array=np.array([0.01,0.5]),
                            rot_dpf_minmax: np.array=np.array([0,5]) 
                            ):
    """Save n optic flow dots files to the current directory"""
    for i in range(n):
        trans_speed=np.min(trans_speed_minmax)+np.random.sample()*trans_speed_minmax.ptp()
        trans_ruf=random_unit_vector()*trans_speed # ruf = right up front
        rot_dpf=np.min(rot_dpf_minmax)+np.random.sample()*rot_dpf_minmax.ptp()
        rot_ruf=random_unit_vector()*rot_dpf
        FHW=optic_flow_dots.generate(trans_ruf=trans_ruf,
                                     rot_ruf=rot_ruf,
                                     n_frames=20,
                                     wid_px=200,
                                     hei_px=200,
                                     )
    
if __name__=="__main__":
    create_optic_flow_files()