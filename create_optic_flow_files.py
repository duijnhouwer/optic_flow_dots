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

def create_optic_flow_files(n=100,
                            trans_speed_minmax: np.array=np.array([0.02,0.02]),
                            rot_dpf_minmax: np.array=np.array([0,0]),
                            savedir: str=os.path.dirname(__file__)+'_dataTEST'):
    start_second=time.time()
    for i in range(n):
        trans_speed=np.min(trans_speed_minmax)+np.random.sample()*trans_speed_minmax.ptp()
        trans_ruf=random_unit_vector(3)*trans_speed # ruf = right up front
        rot_dpf=np.min(rot_dpf_minmax)+np.random.sample()*rot_dpf_minmax.ptp()
        rot_ruf=random_unit_vector(3)*rot_dpf
        # Create the Frame-Height-Width tensor movie
        movie, jumps_px = optic_flow_dots.generate_movie(trans_ruf=trans_ruf,
                                         rot_ruf=rot_ruf,
                                         n_dots=10000,
                                         n_frames=2,
                                         wid_px=300,
                                         hei_px=300,
                                         dot_diam_px=2,
                                         supersample=5)
        # convert to torch tensor
        movie=torch.from_numpy(movie)
        # add a channel dimension with 1 level, meaning grayscale. conv3d needs this dimension
        movie=movie.unsqueeze(0);
        # convert to float32 values, conv3d requires that
        movie=movie.to(torch.float32)
        # save to a file with the 6DOF parameters in the filename
        movie_filename="trxyz=[{:+.9f}_{:+.9f}_{:+.9f}_{:+.9f}_{:+.9f}_{:+.9f}].pt".format(trans_ruf[0],trans_ruf[1],trans_ruf[2],rot_ruf[0],rot_ruf[1],rot_ruf[2])
        movie_filename=os.path.join(savedir, movie_filename)
        torch.save(movie, movie_filename)
        # Also save the jumps_px to a separate file
        jumps_filename="trxyz=[{:+.9f}_{:+.9f}_{:+.9f}_{:+.9f}_{:+.9f}_{:+.9f}]_jumps.pt".format(trans_ruf[0],trans_ruf[1],trans_ruf[2],rot_ruf[0],rot_ruf[1],rot_ruf[2])
        torch.save(jumps_px.to(torch.float32), jumps_filename)
        if i%100==0:
            print("[{}] Saved {} movies to {} (Elapsed time={})".format(inspect.currentframe().f_code.co_name,i,savedir,lgg.format_duration(time.time()-start_second)))
            
    print("[{}] Done".format(inspect.currentframe().f_code.co_name))
    
        
if __name__=="__main__":
    lgg.computer_sleep('prevent')
    create_optic_flow_files()
    lgg.computer_sleep('allow')
      
