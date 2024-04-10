# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 20:29:30 2024

@author: jduij
"""
import numpy as np
import optic_flow_dots
import torch
import os
import inspect

def random_unit_vector():
    V=np.random.sample(3) # create a random 3D vector
    V=V/np.linalg.norm(V) # normalize to unit-length
    V[np.random.sample(V.size)<0.5]*=-1 # randomize sign of each dimension
    return V

def create_optic_flow_files(n=1000,
                            trans_speed_minmax: np.array=np.array([0.1,0.5]),
                            rot_dpf_minmax: np.array=np.array([0,0]),
                            savedir: str="C:\\Users\\jduij\\Documents\\GitHub\\optic_flow_dots_data"
                            ):
    for i in range(1,n+1):
        trans_speed=np.min(trans_speed_minmax)+np.random.sample()*trans_speed_minmax.ptp()
        trans_ruf=random_unit_vector()*trans_speed # ruf = right up front
        rot_dpf=np.min(rot_dpf_minmax)+np.random.sample()*rot_dpf_minmax.ptp()
        rot_ruf=random_unit_vector()*rot_dpf
        # Create the Frame-Height-Width tensor M (movie)
        M=optic_flow_dots.generate(trans_ruf=trans_ruf,
                                     rot_ruf=rot_ruf,
                                     n_frames=25,
                                     wid_px=100,
                                     hei_px=100,
                                     )
        # convert to torch tensor
        M=torch.from_numpy(M)
        # add a channel dimension with 1 level, just because conv3d needs it
        M=M.unsqueeze(0);
        # convert to float32 values, just because conv3d needs it
        M=M.to(torch.float32)
        # save to a file with the 6-DOF parameters in the filename
        fname="trxyz=[{:+.9f}_{:+.9f}_{:+.9f}_{:+.9f}_{:+.9f}_{:+.9f}].pt".format(trans_ruf[0],trans_ruf[1],trans_ruf[2],rot_ruf[0],rot_ruf[1],rot_ruf[2])
        fname=os.path.join(savedir,fname)
        torch.save(M,fname)
        if i%100==0:
            print("[{}] Saved {} files to {}".format(inspect.currentframe().f_code.co_name,i,savedir))
    print("[{}] Done".format(inspect.currentframe().f_code.co_name))
    
        
if __name__=="__main__":
      create_optic_flow_files()