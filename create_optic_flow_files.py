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
import time
import the_luggage as lgg

def random_unit_vector(n: int=3):
    # Generate a random vector with n components
    vector = np.random.randn(n)
    # Normalize the vector
    norm = np.linalg.norm(vector)
    unit_vector = vector / norm
    return unit_vector

def create_optic_flow_files(n=100000,
                            trans_speed_minmax: np.array=np.array([0.02,0.02]),
                            rot_dpf_minmax: np.array=np.array([0,0]),
                            savedir: str=os.path.dirname(__file__)+'_data'):
    start_second=time.time()
    for i in range(1,n+1):
        trans_speed=np.min(trans_speed_minmax)+np.random.sample()*trans_speed_minmax.ptp()
        trans_ruf=random_unit_vector(3)*trans_speed # ruf = right up front
        rot_dpf=np.min(rot_dpf_minmax)+np.random.sample()*rot_dpf_minmax.ptp()
        rot_ruf=random_unit_vector(3)*rot_dpf
        # Create the Frame-Height-Width tensor M (movie)
        M=optic_flow_dots.generate_movie(trans_ruf=trans_ruf,
                                         rot_ruf=rot_ruf,
                                         n_frames=20,
                                         wid_px=200,
                                         hei_px=200,
                                         dot_diam_px=3,
                                         supersample=2)
        # convert to torch tensor
        M=torch.from_numpy(M)
        # add a channel dimension with 1 level, meaning grayscale. conv3d needs this dimension
        M=M.unsqueeze(0);
        # convert to float32 values, conv3d requires that
        M=M.to(torch.float32)
        # save to a file with the 6DOF parameters in the filename
        fname="trxyz=[{:+.9f}_{:+.9f}_{:+.9f}_{:+.9f}_{:+.9f}_{:+.9f}].pt".format(trans_ruf[0],trans_ruf[1],trans_ruf[2],rot_ruf[0],rot_ruf[1],rot_ruf[2])
        fname=os.path.join(savedir,fname)
        torch.save(M,fname)
        if i==1 or i%100==0:
            print("[{}] Saved {} files to {} (Elapsed time={})".format(inspect.currentframe().f_code.co_name,i,savedir,lgg.format_duration(time.time()-start_second)))
            
    print("[{}] Done".format(inspect.currentframe().f_code.co_name))
    
        
if __name__=="__main__":
      create_optic_flow_files()
      
