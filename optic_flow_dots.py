# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 20:48:00 2024

@author: jduij
"""
import numpy as np
import scipy
from scipy.signal import convolve2d


def create_antialiased_frame(width=200, height=200, dotdiam=3, supersample=4, xy=300):
    # Initialize the super-sampled image
    super_width = width * supersample
    super_height = height * supersample
    image = np.zeros((super_height, super_width))

    # Determine coordinates
    if np.isscalar(xy):
        # Generate a 2 by xy matrix of random coordinates
        xy = np.random.randint(0, high=min(super_width, super_height), size=(2, xy))
    elif xy.shape[0]==2:
        xy = xy * supersample
        xy = np.round(xy).astype(int)
        valid_mask = (xy[0] >= 0) & (xy[0] < super_width) & (xy[1] >= 0) & (xy[1] < super_height)
        xy = xy[:, valid_mask]
    else: 
        raise Exception("xy must be a scalar (N) or a 2xN matrix") 

    # Set the specified pixels to 1
    image[xy[0], xy[1]] = 1

    # Create a circular disk kernel
    radius = (dotdiam * supersample) // 2
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    disk = x**2 + y**2 <= radius**2
    disk = disk.astype(float)

    # Convolve the image with the disk
    convolved_image = convolve2d(image, disk, mode='same', boundary='fill', fillvalue=0)

    # Threshold the image
    thresholded_image = (convolved_image > 0).astype(int)

    # Downsample the image using a supersample by supersample averaging kernel
    kernel = np.ones((supersample, supersample)) / (supersample**2)
    downsampled_image = convolve2d(thresholded_image, kernel, mode='valid')[::supersample, ::supersample]
    return downsampled_image
   
    
def generate_movie(n_dots:int=5000,
                   trans_ruf: np.array=np.array([0, 0.0, 0.02]), # right up front, 2-norm is translation speed in units per frame
                   rot_ruf=np.array([0, 0, 0]), # right up front, 2-norm is rotation speed in degrees per frame
                   n_frames: int=100,
                   wid_px: int=200,
                   hei_px: int=200,
                   dot_diam_px: int=5,
                   supersample: int=4,
                   dot_life_fr: int=0,
                   dot_life_sync: bool=False,
                   ): 
    
    # allocate the output with dimensions FrameNr, Height, Width
    output_fhw=np.zeros((n_frames,hei_px,wid_px));
    
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
        
        XY= np.stack((X*wid_px, Y*hei_px), axis=0)
        output_fhw[fr,:,:] = create_antialiased_frame(width=wid_px, height=hei_px, dotdiam=dot_diam_px, supersample=supersample, xy=XY)
        
    return output_fhw
    

if __name__ == "__main__":
    M=generate_movie() 
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    