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
from multiprocessing import Pool


def random_unit_vector(n: int=3):
    # Generate a random vector with n components
    vector = np.random.randn(n)
    # Normalize the vector
    norm = np.linalg.norm(vector)
    unit_vector = vector / norm
    return unit_vector


def create_optic_flow_files(n: int=100000,
                            trans_speed_minmax: np.array=np.array([0.02,0.02]),
                            rot_dpf_minmax: np.array=np.array([0,0]),
                            savedir: str='S:\optic_flow_dots_data'): # str=os.path.dirname(__file__)+'_dataTEST'
    start_second=time.time()
    
    print(f'N={n}')
    for i in range(n):
        # Pick a rotation and translation speed
        trans_speed = np.min(trans_speed_minmax) + np.random.sample() * trans_speed_minmax.ptp()
        rot_dpf = np.min(rot_dpf_minmax) + np.random.sample() * rot_dpf_minmax.ptp()
        
        # Create the translation and rotation vectors 
        trans_ruf = random_unit_vector(3) * trans_speed # ruf: right up front
        rot_ruf = random_unit_vector(3) * rot_dpf
        
        # Create the Frame-Height-Width tensor movie
        movie = optic_flow_dots.generate_movie(trans_ruf=trans_ruf,
                                                rot_ruf=rot_ruf,
                                                n_dots=10000,
                                                n_frames=2,
                                                wid_px=300,
                                                hei_px=300,
                                                dot_diam_px=3,
                                                supersample=4)
        # convert to torch tensor
        movie=torch.from_numpy(movie)
        # add a channel dimension with 1 level, meaning grayscale. nn.Conv3d needs this dimension
        movie=movie.unsqueeze(0);
        # Convert to uint8 to save storage space and, hopefully, reduce loading 
        # time enough to offset the additional work of converting it back to 
        # float32 when we load the movie using the DataLoader
        movie=movie*255
        movie=movie.to(torch.uint8)
        
        # save to a file with the 6DOF parameters in the filename
        movie_filename="trxyz=[{:+.9f}_{:+.9f}_{:+.9f}_{:+.9f}_{:+.9f}_{:+.9f}].pt".format(trans_ruf[0],trans_ruf[1],trans_ruf[2],rot_ruf[0],rot_ruf[1],rot_ruf[2])
        movie_filename=os.path.join(savedir, movie_filename)
        try:
            torch.save(movie, movie_filename)
        except:
            raise Exception(f"Could not save '{movie_filename}' for unknown reason. Maybe another process was writing a file with the same name at the same time? No biggie, just one of a million")
        
        if i%100==0:
            print("[{}] Saved {} movies to {} (Elapsed time={})".format(inspect.currentframe().f_code.co_name,i,savedir,lgg.format_duration(time.time()-start_second)))
            
    print("[{}] Done".format(inspect.currentframe().f_code.co_name))
    
        
if __name__=="__main__":
    lgg.computer_sleep('disable')
    try:
        n_workers = 6 
        with Pool(n_workers) as p:
            print("Start pool")
            result = p.imap(create_optic_flow_files, [100000] *n_workers)
            print("Start timer")
            time.sleep(3600*24*7)
            print("Stop timer")    
    except KeyboardInterrupt:
        print("[{}] Interupted with CTRL-C".format(inspect.currentframe().f_code.co_name))
    except: 
        raise Exception('Something happened and the processes stopped')
    finally:
        lgg.computer_sleep('enable')
        p.terminate()
        p.join()
        print('[-: The End :-]')