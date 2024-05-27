# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 20:48:00 2024

@author: jduij
"""
import numpy as np
import scipy

def random_unit_vector(n: int=3):
    # Generate a random vector with n components
    vector = np.random.randn(n)
    # Normalize the vector
    norm = np.linalg.norm(vector)
    unit_vector = vector / norm
    return unit_vector

def create_random_point():
    # Generate the random values between -1 and 1 for the first three elements
    random_values = np.random.uniform(-1, 1, 3)
    # Append 1 as the fourth element
    vector = np.append(random_values, 1)
    # Convert to a column vector
    column_vector = vector.reshape(-1, 1)
    return column_vector

def transform_point(M4x4, xyz1):
    # Ensure xyz1 is a column vector
    xyz1 = np.reshape(xyz1, (3, 1))
    # Ensure xyz1 is a 4-element homogeneous coordinate by appending 1
    xyz1_homogeneous = np.vstack((xyz1, [[1]]))
    # Apply the transformation matrix to the point
    xyz2_homogeneous = np.dot(M4x4, xyz1_homogeneous)
    # Convert back to 3-element vector by dividing by the homogeneous coordinate
    xyz2 = xyz2_homogeneous[:3] / xyz2_homogeneous[3]
    return xyz2.flatten()


def generate_signal_array(n_dots: int=1000,
                   trans_ruf: np.array=np.array([0.0, 0.0, 0.02]), # right up front, 2-norm is translation speed in units per frame
                   rot_ruf: np.array=np.array([0.0, 0.0, 0.0]), # right up front, 2-norm is rotation speed in degrees per frame
                   ):

    # create the transformation matrix M4x4
    M4x4 = np.eye(4,4)
    M4x4[0:3,3] = trans_ruf * np.array([-1.0,1.0,-1.0])
    M4x4[0:3,0:3] = scipy.spatial.transform.Rotation.from_rotvec(rot_ruf, degrees=True).as_matrix()

    # Pre-allocate the return matrix
    signal_u_v_du_dv = np.full((4, n_dots), np.nan)

    n = 0
    while n < n_dots:
        A = np.random.uniform(-1, 1, 3)
        # Make sure the in front
        A[1] = abs(A[1])
        # check that the point is in the unit sphere
        if np.linalg.norm(A)>1:
            continue
        # convert points A to image plane coordinate
        Au = A[0]/A[2]
        Av = A[1]/A[2]
        # check that point A is inside the 45 degree radius viewport
        if np.hypot(Au,Av) > np.sqrt(2)/2:
            continue

        # Move the dot from point A to B according to the flow
        B = transform_point(M4x4, A)
        # check that the point is in the unit sphere
        if np.linalg.norm(B)>1:
            continue
        # convert points B to image plane coordinate
        Bu = B[0]/B[2]
        Bv = B[1]/B[2]
        # check that point A is inside the 45 degree radius viewport
        if np.hypot(Bu,Bv) > np.sqrt(2)/2:
            continue

        # We found the next valid point!
        signal_u_v_du_dv[0][n] = Au
        signal_u_v_du_dv[1][n] = Av
        signal_u_v_du_dv[2][n] = Bu-Au
        signal_u_v_du_dv[3][n] = Bv-Av
        n += 1

    return signal_u_v_du_dv



if __name__ == "__main__":
    S = generate_signal_array()
