# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 20:48:00 2024

@author: jduij
"""
import numpy as np
import scipy
from scipy.signal import convolve2d

def random_unit_vector(n: int=3):
    # Generate a random vector with n components
    vector = np.random.randn(n)
    # Normalize the vector
    norm = np.linalg.norm(vector)
    unit_vector = vector / norm
    return unit_vector


def create_antialiased_frame(width=200, height=200, dotdiam=3, supersample=4, xy=300):

    # xy is assumed to be unit coordinates. xy-points with coordinates
    # outside [0..1] will not be rendered

    # Initialize the super-sampled image
    super_width = width * supersample
    super_height = height * supersample
    image = np.zeros((super_height, super_width))

    if np.isscalar(xy):
        # Generate a 2 by xy matrix of random unit coordinates
        n = xy
        xy = np.random.rand(2, n)

    xy[0] *= super_width
    xy[1] *= super_height
    xy = np.round(xy).astype(int)
    valid_mask = (xy[0] >= 0) & (xy[0] < super_width) & (xy[1] >= 0) & (xy[1] < super_height)
    xy = xy[:, valid_mask]

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


def generate_movie(n_dots: int=8000,
                   trans_ruf: np.array=np.array([0.0, 0.0, 0.02]), # right up front, 2-norm is translation speed in units per frame
                   rot_ruf: np.array=np.array([0.0, 0.0, 0.0]), # right up front, 2-norm is rotation speed in degrees per frame
                   n_frames: int=2,
                   wid_px: int=300,
                   hei_px: int=300,
                   dot_diam_px: int=2,
                   supersample: int=5,
                   ):

    # Allocate the movie tensor with dimensions FrameNr, Height, Width
    movie_fhw=np.zeros((n_frames,hei_px,wid_px), dtype=np.float32);

    # Give the dots starting positions in a 2x2x2 box with the origin at the center
    dots_xyz = np.random.random_sample((3,n_dots))*2.0-1.0;

    # create the transformation matrix M4x4
    M4x4 = np.eye(4,4)
    M4x4[0:3,3] = trans_ruf * np.array([-1.0,1.0,-1.0])
    M4x4[0:3,0:3] = scipy.spatial.transform.Rotation.from_rotvec(rot_ruf, degrees=True).as_matrix()


    # loop over the frames to be rendered
    for fr in range(n_frames):

        dots_xyz = np.matmul(M4x4, np.concatenate((dots_xyz, np.ones((1,n_dots))), axis=0))[0:3,:]
        too_neg = dots_xyz<-1
        too_pos = dots_xyz>1
        # Create a logical index tensor to all 4 coordinates of dots that have at least one out of bounds coordinate
        out_of_bound_dots_idx = np.tile(np.any(too_neg|too_pos, axis=0),(3,1))
        # Of the out of bounds dots, replace the coordinates that were not themselves out of bounds with fresh random values
        idx = out_of_bound_dots_idx & ~too_neg & ~too_pos
        dots_xyz[idx] = np.random.random_sample(np.sum(idx))*2.0-1.0
        # Wrap the out of bound coordinates around
        dots_xyz[too_neg] += 2
        dots_xyz[too_pos] -= 2

        # Detect which dots are in unit sphere and in front of camera
        in_sphere = np.linalg.norm(dots_xyz, axis=0)<=1
        in_front = dots_xyz[2,:]>0
        in_frontal_half_dome = np.logical_and(in_sphere, in_front)

        X = np.divide(dots_xyz[0,in_frontal_half_dome], dots_xyz[2,in_frontal_half_dome])
        Y = np.divide(dots_xyz[1,in_frontal_half_dome], dots_xyz[2,in_frontal_half_dome])

        # Limit to 90 x 90 degree viewport
        edgeval = np.sin(45/180*np.pi);
        keep_idx = np.logical_and(abs(X)<edgeval, abs(Y)<edgeval)
        X = X[keep_idx]
        Y = Y[keep_idx]

        # Normalize image plane dimensions to [0 1)
        X += edgeval
        Y += edgeval
        X /= 2 * edgeval
        Y /= 2 * edgeval

        if n_frames==2 and n_dots==1:
            # Special when n_frames==2 and n_dots==1: log the displacement of the single dot in pixels
            if X.size==0 or Y.size==0:
                return None
            elif fr==0:
                X0 = X * wid_px
                Y0 = Y * hei_px
            elif fr==1:
                X1 = X * wid_px
                Y1 = Y * hei_px
                return np.hypot(X1-X0, Y1-Y0)
        else:
            # Append an antialiased frame to the movie
            XY= np.stack((X, Y), axis=0)
            movie_fhw[fr,:,:] = create_antialiased_frame(wid_px, hei_px, dot_diam_px, supersample, XY)

    return movie_fhw






if __name__ == "__main__":
    M=generate_movie()
