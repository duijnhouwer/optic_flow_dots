# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 20:48:00 2024

@author: jduij
"""
import numpy as np
import scipy
import matplotlib.pyplot as plt
import time
   
def generate(n_dots:int=5000,
                    trans_ruf: np.array=np.array([0, 0.0, 0.02]), # right up front, 2-norm is translation speed in units per frame
                    rot_ruf=np.array([0, 0, 0]), # right up front, 2-norm is rotation speed in degrees per frame
                    n_frames:int=100,
                    wid_px:int=200,
                    hei_px:int=200,
                    dot_life_fr:int=0,
                    dot_life_sync:bool=False,
                    show_wait_s:float=-1 # negative means don't show the animation 
                    ): 
    
    # allocate the output with dimensions FrameNr, Height, Width
    output_fhw=np.zeros((n_frames,hei_px,wid_px),bool);
    
    # Give the dots starting positions in a 2x2x2 box with the origin at the center
    dots_xyz = np.random.random_sample((3,n_dots))*2.0-1.0;
    
    
    # If the dots are to have limited lifetime, set their remaining frames now
    #if dot_life_fr > 0 and not dot_life_sync:
    #    dots_frames_left = np.random.randint(dot_life_fr, size=n_dots)
    
    # create the transformation matrix
    M = np.eye(4,4)
    M[0:3,3]=trans_ruf*[-1,1,-1];
    M[0:3,0:3]=scipy.spatial.transform.Rotation.from_rotvec(rot_ruf,degrees=True).as_matrix()
    
    # loop over the frames to be rendered
    for fr in range(n_frames):
    
        dots_xyz=np.matmul(M,np.concatenate((dots_xyz, np.ones((1,n_dots))),axis=0))[0:3,:]
        too_neg=dots_xyz<-1
        too_pos=dots_xyz>1
        # create a logical index tensor to all 4 coordinates of dots that have at least one out of bounds coordinate
        out_of_bound_dots_idx=np.tile(np.any(too_neg|too_pos,axis=0),(3,1))
        # of the out of bounds dots, replace the coordinates that were not themselves out of bounds with fresh random values
        idx=out_of_bound_dots_idx & ~too_neg & ~too_pos
        dots_xyz[idx]=np.random.random_sample(np.sum(idx))*2.0-1.0
        # Wrap the out of bound coordinates around
        dots_xyz[too_neg]+=2
        dots_xyz[too_pos]-=2
        #breakpoint()
        
        # Detect which dots are in unit sphere and in front of camera 
        in_sphere=np.linalg.norm(dots_xyz, axis=0)<=1
        
       
        in_front=dots_xyz[2,:]>0
        in_frontal_half_dome=np.logical_and(in_sphere,in_front)
        X = np.divide(dots_xyz[0,in_frontal_half_dome],dots_xyz[2,in_frontal_half_dome])
        Y = np.divide(dots_xyz[1,in_frontal_half_dome],dots_xyz[2,in_frontal_half_dome])
        # limit to 90 x 90 degree viewport
        edgeval=np.sin(45/180*np.pi);
        keep_idx=np.logical_and(abs(X)<edgeval,abs(Y)<edgeval)
        X=X[keep_idx]
        Y=Y[keep_idx]
        # Normalize dimensions to [0 1)
        X+=edgeval
        Y+=edgeval
        X/=2*edgeval
        Y/=2*edgeval
        # set the pixels of the current frame in output_fhw
        output_fhw[fr,(Y*hei_px).astype(int),(X*wid_px).astype(int)]=True;      
        
        if show_wait_s>=0:
            plt.imshow(output_fhw[fr,:,:].astype(float),cmap='gray', vmin=0.0, vmax=1.0)
            plt.show(block=False)
            time.sleep(show_wait_s);
        
    return output_fhw
    

if __name__ == "__main__":
    M=generate() 
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    