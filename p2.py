import numpy as np
import matplotlib.pyplot as plt
from p1 import *
from epipolar_utils import *

'''
COMPUTE_EPIPOLE computes the epipole e in homogenous coordinates
given the fundamental matrix
Arguments:
    F - the Fundamental matrix solved for with normalized_eight_point_alg(points1, points2)

Returns:
    epipole - the homogenous coordinates [x y 1] of the epipole in the image
'''
def compute_epipole(F):
    # TODO: Implement this method!
    
    U, S, Vh = np.linalg.svd(F.T, full_matrices = True)
    
    return (Vh[2,:]/Vh[2,2])

    raise Exception('Not Implemented Error')

'''
COMPUTE_H computes a homography to map an epipole to infinity along the horizontal axis 
Arguments:
    e - the epipole
    im2 - the image
Returns:
    H - homography matrix
'''
def compute_H(e, im):
    # TODO: Implement this method!
   
    width,height =im.shape

    T = [[1,0,-width/2],[0,1,-height/2],[0,0,1]]
    Te = np.matmul(T,e)
    Te=Te/Te[2]

    e1= Te[0]
    e2 = Te[1]
    e_mod = np.sqrt(e1*e1 + e2*e2)
    if(e1 >= 1):
        a = 1
    else:
        a = -1
    R = [[a*e1/e_mod, a*e2/e_mod, 0],[-a*e2/e_mod, a*e1/e_mod, 0],[0,0,1]]
    RTe = np.matmul(R,Te)
    G =[[1,0,0],[0,1,0],[-1/RTe[0],0,1]]

    GRTe = np.matmul(G,RTe.T)
    H = np.matmul(R,T)
    
    H = np.matmul(G,H)
    
    inv_T = np.linalg.inv(T)

    H = np.matmul(inv_T,H)
    
    return H
    raise Exception('Not Implemented Error')

'''
COMPUTE_MATCHING_HOMOGRAPHIES determines homographies H1 and H2 such that they
rectify a pair of images
Arguments:
    e2 - the second epipole
    F - the Fundamental matrix
    im2 - the second image
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
Returns:
    H1 - the homography associated with the first image
    H2 - the homography associated with the second image
'''
def compute_matching_homographies(e2, F, im2, points1, points2):
    # TODO: Implement this method!
    
    #points1[:,[0,1]] = points1[:,[1,0]]
    #points2[:,[0,1]] = points2[:,[1,0]]
    H2 = np.asmatrix(compute_H(e2, im2))
    v = [1,1,1]
    e_cross = [[0, -e2[2], e2[1]],[e2[2], 0, -e2[0]],[-e2[1],e2[0], 0]]
    e2 = np.asmatrix(e2)
    v = np.asmatrix(v)
    N = np.matmul(e2.T, v)
    M = np.matmul(e_cross, F)
    M = np.add(M,N)
    points1_cap = (np.matmul(H2, np.matmul(M, points1.T)))
    points2_cap = (np.matmul(H2, points2.T))
    #print(points1_cap.shape, points2_cap.shape)
    #print("M",M)
    row, column = points2_cap.shape
    for i in range(column):
        p1 = points1_cap[2,i]
        p2 = points2_cap[2,i]
        points1_cap[0,i] = points1_cap[0,i]/p1
        points1_cap[1,i] = points1_cap[1,i]/p1
        points1_cap[2,i] = points1_cap[2,i]/p1
        points2_cap[0,i] = points2_cap[0,i]/p2
        points2_cap[1,i] = points2_cap[1,i]/p2
        points2_cap[2,i] = points2_cap[2,i]/p2
    
    W = points1_cap.T 
    #print(points1_cap.shape)
    b = points2_cap[0,:]
    #print("b",b.shape)
    U, S,Vh = np.linalg.svd(W, full_matrices=False)
    #print("U",U.shape,Vh.shape)
    row = S.size
    #print("W,",W)
    #print(S)
   
    l = np.identity(row)
    for i in range(row):
        l[i,i] = S[i]
    #print(l,np.linalg.inv(l))
    a = np.matmul(U.T,b.T)
    #print("Teste", np.linalg.inv(l).shape, a.shape,Vh.shape)
    a = np.matmul(np.linalg.inv(l), a)
    #print("Test", a)
    a = np.matmul(Vh.T,a)
    
    Ha = np.identity(3)
    Ha[0,0] = a[0]
    Ha[0,1] = a[1]
    Ha[0,2] = a[2]
    
    H1 = np.matmul(H2,M)
    H1 = np.asarray(np.matmul(Ha,H1))
    H2 =np.asarray(H2)
    #print (M, Ha, H1,H2)
    #print(H1)
    return (H1,H2)
    raise Exception('Not Implemented Error')

if __name__ == '__main__':
    # Read in the data
    im_set = 'data/set1'
    im1 = imread(im_set+'/image1.jpg')
    im2 = imread(im_set+'/image2.jpg')
    points1 = get_data_from_txt_file(im_set+'/pt_2D_1.txt')
    points2 = get_data_from_txt_file(im_set+'/pt_2D_2.txt')
    assert (points1.shape == points2.shape)

    F = normalized_eight_point_alg(points1, points2)
    # F is such that such that (points2)^T * F * points1 = 0, so e1 is e' and e2 is e
    e1 = compute_epipole(F.T)
    e2 = compute_epipole(F)
    print("e1", e1)
    print("e2", e2)

    # Find the homographies needed to rectify the pair of images
    H1, H2 = compute_matching_homographies(e2, F, im2, points1, points2)
    print('')

    # Transforming the images by the homographies
    new_points1 = H1.dot(points1.T)
    new_points2 = H2.dot(points2.T)

    new_points1 /= new_points1[2,:]
    new_points2 /= new_points2[2,:]
    new_points1 = new_points1.T
    new_points2 = new_points2.T

    rectified_im1, offset1 = compute_rectified_image(im1, H1)
    rectified_im2, offset2 = compute_rectified_image(im2, H2)
    new_points1 -= offset1 + (0,)
    new_points2 -= offset2 + (0,)

    # Plotting the image
    F_new = normalized_eight_point_alg(new_points1, new_points2)
    plot_epipolar_lines_on_images(new_points1, new_points2, rectified_im1, rectified_im2, F_new)
    plt.show()
