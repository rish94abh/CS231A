import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from epipolar_utils import *

'''
LLS_EIGHT_POINT_ALG  computes the fundamental matrix from matching points using 
linear least squares eight point algorithm
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    F - the fundamental matrix such that (points2)^T * F * points1 = 0
Please see lecture notes and slides to see how the linear least squares eight
point algorithm works
'''


def lls_eight_point_alg(points1, points2):    
    # TODO: Implement this method!
    rows, columns = points1.shape
    points1 = np.asmatrix(points1)
    points2 = np.asmatrix(points2)
    
    points1[:,(0,1)] = points1[:,(1,0)]
    points2[:,(0,1)] = points2[:,(1,0)]
    W = np.zeros((rows,9))
    for i in range(rows):
        W[i,0] =  points2[i,1]*points1[i,1]
        W[i,1] =  points1[i,0]*points2[i,1]
        W[i,2] =  points2[i,1]
        W[i,3] =  points1[i,1]*points2[i,0] 
        W[i,4] =  points2[i,0]*points1[i,0]
        W[i,5] =  points2[i,0]
        W[i,6] =  points1[i,1]
        W[i,7] =  points1[i,0]
        W[i,8] =  1

    
    U, S, Vh = np.linalg.svd(W, full_matrices=True)
    f = Vh[8,:].reshape(3,3)
    U_f, S_f,Vh_f = np.linalg.svd(f, full_matrices=True)
    I = np.zeros((3,3))
    I[0][0] = S_f[0]
    I[1][1] = S_f[1]
    f =np.matmul(U_f,np.matmul(I,Vh_f))

    return (f)
    raise Exception('Not Implemented Error')

'''
NORMALIZED_EIGHT_POINT_ALG  computes the fundamental matrix from matching points
using the normalized eight point algorithm
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    F - the fundamental matrix such that (points2)^T * F * points1 = 0
Please see lecture notes and slides to see how the normalized eight
point algorithm works
'''
def normalized_eight_point_alg(points1, points2):
    # TODO: Implement this method!
    
    N,M = points1.shape
    
    norm_points1 = norm_points2 = np.zeros((N,3))
    point1_centroid = point2_centroid = np.zeros((1,3))    
    s1 = s2 = 0
    residual_p1 = residual_p2 = np.zeros((N,3))

    
    #Translation
    point1_centroid = np.asmatrix(points1).mean(axis = 0)
    point2_centroid = np.asmatrix(points2).mean(axis = 0)
    
    x1=point1_centroid[0,0]*np.ones(N)
    x2=point2_centroid[0,0]*np.ones(N)
    y1 =point1_centroid[0,1]*np.ones(N)
    y2=point2_centroid[0,1]*np.ones(N)
    z1=z2 =np.ones(N)
    
    p1 = np.transpose(np.vstack([x1,y1,z1]))
    p2 =np.transpose(np.vstack([x2,y2,z2]))

    residual_p1 = points1 - p1

    residual_p2 = points2 - p2  

    #Scaling
    for i in range(N):        
        s1 = s1 + (residual_p1[i,0]*residual_p1[i,0]+ residual_p1[i,1]*residual_p1[i,1])
        s2 = s2 + (residual_p2[i,0]*residual_p2[i,0]+ residual_p2[i,1]*residual_p2[i,1])
        
    s1 = np.sqrt((2*N)/s1)  
    s2 = np.sqrt((2*N)/s2)

    
    T = [[s1,0,-point1_centroid[0,0]*s1],[0,s1,-point1_centroid[0,1]*s1],[0,0,1]]
    Tp = [[s2,0,-point2_centroid[0,0]*s2],[0,s2,-point2_centroid[0,1]*s2],[0,0,1]]


    norm_points1 = residual_p1*s1
    norm_points2 = residual_p2*s2
    norm_points1[:,2] = norm_points2[:,2] = 1

    F = lls_eight_point_alg(norm_points1, norm_points2)
    F = np.matmul(F, T)
    F = np.matmul(np.transpose(Tp),F)
    
    return F
    raise Exception('Not Implemented Error')

'''
PLOT_EPIPOLAR_LINES_ON_IMAGES given a pair of images and corresponding points,
draws the epipolar lines on the images
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    im1 - a HxW(xC) matrix that contains pixel values from the first image 
    im2 - a HxW(xC) matrix that contains pixel values from the second image 
    F - the fundamental matrix such that (points2)^T * F * points1 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    Nothing; instead, plots the two images with the matching points and
    their corresponding epipolar lines. See Figure 1 within the problem set
    handout for an example
'''
def plot_epipolar_lines_on_images(points1, points2, im1, im2, F):

    def plot_epipolar_lines_on_image(points1, points2, im, F):
        im_height = im.shape[0]
        im_width = im.shape[1]
        lines = F.T.dot(points2.T)
        plt.imshow(im, cmap='gray')
        for line in lines.T:
            a,b,c = line
            xs = [1, im.shape[1]-1]
            ys = [(-c-a*x)/b for x in xs]
            plt.plot(xs, ys, 'r')
        for i in range(points1.shape[0]):
            x,y,_ = points1[i]
            plt.plot(x, y, '*b')
        plt.axis([0, im_width, im_height, 0])

    # We change the figsize because matplotlib has weird behavior when 
    # plotting images of different sizes next to each other. This
    # fix should be changed to something more robust.
    new_figsize = (8 * (float(max(im1.shape[1], im2.shape[1])) / min(im1.shape[1], im2.shape[1]))**2 , 6)
    fig = plt.figure(figsize=new_figsize)
    plt.subplot(121)
    plot_epipolar_lines_on_image(points1, points2, im1, F)
    plt.axis('off')
    plt.subplot(122)
    plot_epipolar_lines_on_image(points2, points1, im2, F.T)
    plt.axis('off')

'''
COMPUTE_DISTANCE_TO_EPIPOLAR_LINES  computes the average distance of a set a 
points to their corresponding epipolar lines. Compute just the average distance
from points1 to their corresponding epipolar lines (which you get from points2).
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    F - the fundamental matrix such that (points2)^T * F * points1 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    average_distance - the average distance of each point to the epipolar line
'''
def compute_distance_to_epipolar_lines(points1, points2, F):
    # TODO: Implement this method!
    
    rows,column = points1.shape
    #points1[:,[0,1]] = points1[:,[1,0]]
    #points2[:,[0,1]] = points2[:,[1,0]]
    distance = np.zeros((rows,rows))

    l = np.zeros((3,rows))
    #d = np.zeros(rows)
    l = np.matmul(F,points2.T)
    
    #print("lesss",points1)
    for i in range(rows):
        p =l[1,i]
        points1[i,1] = points1[i,1] + l[2,i]/p
    #print(l.shape, points1.shape)
    l[2,:]=0
    for i in range(rows):
        l[0,i] = l[0,i]/np.sqrt(l[0,i]**2+l[1,i]**2)
        l[1,i] = l[1,i]/np.sqrt(l[0,i]**2+l[1,i]**2)


    distance = (np.absolute(np.matmul(points1,l )))

    d = (np.trace(distance))

    return d
    
    raise Exception('Not Implemented Error')

if __name__ == '__main__':
    for im_set in ['data/set1', 'data/set2']:
        print('-'*80)
        print("Set:", im_set)
        print('-'*80)

        # Read in the data
        im1 = imread(im_set+'/image1.jpg')
        im2 = imread(im_set+'/image2.jpg')
        points1 = get_data_from_txt_file(im_set+'/pt_2D_1.txt')
        points2 = get_data_from_txt_file(im_set+'/pt_2D_2.txt')
        assert (points1.shape == points2.shape)

        # Running the linear least squares eight point algorithm
        F_lls = lls_eight_point_alg(points1, points2)
        print("Fundamental Matrix from LLS  8-point algorithm:\n", F_lls)
        print("Distance to lines in image 1 for LLS:", \
            compute_distance_to_epipolar_lines(points1, points2, F_lls))
        print("Distance to lines in image 2 for LLS:", \
            compute_distance_to_epipolar_lines(points2, points1, F_lls.T))

        # Running the normalized eight point algorithm
        F_normalized = normalized_eight_point_alg(points1, points2)

        pFp = [points2[i].dot(F_normalized.dot(points1[i])) 
            for i in range(points1.shape[0])]
        print("p'^T F p =", np.abs(pFp).max())
        print("Fundamental Matrix from normalized 8-point algorithm:\n", \
            F_normalized)
        print("Distance to lines in image 1 for normalized:", \
            compute_distance_to_epipolar_lines(points1, points2, F_normalized))
        print("Distance to lines in image 2 for normalized:", \
            compute_distance_to_epipolar_lines(points2, points1, F_normalized.T))

        # Plotting the epipolar lines
        plot_epipolar_lines_on_images(points1, points2, im1, im2, F_lls)
        plot_epipolar_lines_on_images(points1, points2, im1, im2, F_normalized)

        plt.show()
