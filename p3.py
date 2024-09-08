import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import scipy.io as sio
import matplotlib.gridspec as gridspec
from epipolar_utils import *

'''
FACTORIZATION_METHOD The Tomasi and Kanade Factorization Method to determine
the 3D structure of the scene and the motion of the cameras.
Arguments:
    points_im1 - N points in the first image that match with points_im2
    points_im2 - N points in the second image that match with points_im1

    Both points_im1 and points_im2 are from the get_data_from_txt_file() method
Returns:
    structure - the structure matrix
    motion - the motion matrix
'''
def factorization_method(points_im1, points_im2):
    # TODO: Implement this method!

    #centering
    row, column = points_im1.shape
    print(points_im1.shape)
    points_im1 =np.asmatrix(points_im1)
    points_im2 =np.asmatrix(points_im2)
    
    points_im1[:,[0,1]] = points_im1[:,[1,0]]
    points_im2[:,[0,1]] = points_im2[:,[1,0]]
    
    points_im1_center = points_im1 - (np.matrix(points_im1).mean(axis = 0))
    points_im2_center = points_im2 - (np.matrix(points_im2).mean(axis = 0))
    
    points_im1_center = points_im1_center [:,:-1]
    points_im2_center = points_im2_center [:,:-1]
    #points_im1_center[:,[0,1]] = points_im1_center[:,[1,0]]
    #points_im2_center[:,[0,1]] = points_im2_center[:,[1,0]]
    #D = np.zeros((row,4))
    D = np.zeros((row,4))
    D = np.concatenate((points_im1_center, points_im2_center),axis = 1)
    
    D = np.transpose(D)
    

    U, W, Vh = np.linalg.svd(D, full_matrices=False)
    
    U = U[:,0:3] 
    #W = W[0:3]
    #Vh = Vh[0:3,:]
    print(W)
    Diag = np.eye(4)
    Diag[0][0] = W[0]
    Diag[1][1] = W[1]
    Diag[2][2] = W[2]
    Diag[3][3] = W[3]

    #structure = (Diag.dot(Vh[0:3,:]))
    structure = np.matmul(Diag[0:3,0:3],Vh[0:3,:])
    #print(np.shape(structure))
    #structure[[0,1],:] = structure[[1,0],:]
    #structure[[1,2],:] = structure[[2,1],:]
    #structure[2,:] = -1*structure[2,:]
    
    structure = np.asarray((structure))/10
    U = np.asarray(U)

    return(structure*10, U)
    raise Exception('Not Implemented Error')

if __name__ == '__main__':
    for im_set in ['data/set1', 'data/set1_subset']:
        # Read in the data
        im1 = imread(im_set+'/image1.jpg')
        im2 = imread(im_set+'/image2.jpg')
        points_im1 = get_data_from_txt_file(im_set + '/pt_2D_1.txt')
        points_im2 = get_data_from_txt_file(im_set + '/pt_2D_2.txt')
        points_3d = get_data_from_txt_file(im_set + '/pt_3D.txt')
        assert (points_im1.shape == points_im2.shape)

        # Run the Factorization Method
        structure, motion = factorization_method(points_im1, points_im2)

        # Plot the structure
        fig = plt.figure()
        ax = fig.add_subplot(121, projection = '3d')
        scatter_3D_axis_equal(structure[0,:], structure[1,:], structure[2,:], ax)
        ax.set_title('Factorization Method')
        ax = fig.add_subplot(122, projection = '3d')
        scatter_3D_axis_equal(points_3d[:,0], points_3d[:,1], points_3d[:,2], ax)
        ax.set_title('Ground Truth')

        plt.show()
