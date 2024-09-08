# CS231A Homework 1, Problem 2
import numpy as np

'''
DATA FORMAT

In this problem, we provide and load the data for you. Recall that in the original
problem statement, there exists a grid of black squares on a white background. We
know how these black squares are setup, and thus can determine the locations of
specific points on the grid (namely the corners). We also have images taken of the
grid at a front image (where Z = 0) and a back image (where Z = 150). The data we
load for you consists of three parts: real_XY, front_image, and back_image. For a
corner (0,0), we may see it at the (137, 44) pixel in the front image and the
(148, 22) pixel in the back image. Thus, one row of real_XY will contain the numpy
array [0, 0], corresponding to the real XY location (0, 0). The matching row in
front_image will contain [137, 44] and the matching row in back_image will contain
[148, 22]
'''

'''
COMPUTE_CAMERA_MATRIX
Arguments:
     real_XY - Each row corresponds to an actual point on the 2D plane
     front_image - Each row is the pixel location in the front image where Z=0
     back_image - Each row is the pixel location in the back image where Z=150
Returns:
    camera_matrix - The calibrated camera matrix (3x4 matrix)
'''
def compute_camera_matrix(real_XY, front_image, back_image):
    # TODO: Fill in this code
    # Hint: reshape your values such that you have PM=p,
    # and use np.linalg.lstsq or np.linalg.pinv to solve for M.
    # See https://apimirror.com/numpy~1.11/generated/numpy.linalg.pinv
    #
    # Our solution has shapes for M=(8,), P=(48,8), and p=(48,)
    # Alternatively, you can set things up such that M=(4,2), P=(24,4), and p=(24,2)
    # Lastly, reshape and add the (0,0,0,1) row to M to have it be (3,4)

    # BEGIN YOUR CODE HERE
    Z = np.zeros((24,1))
    Z=Z+1
    #front_image = np.append(front_image, Z, axis = 1)
    #Z = Z + 150
    #back_image = np.append(back_image, Z, axis = 1)
    
    P = np.append(front_image, back_image, axis = 0)
    #P = np.append(P, Z, axis = 1)
    Y = np.zeros((12,1)) 
    p1 = np.append(real_XY, Y, axis = 1)
    Y = Y+150
    p2 = np.append(real_XY, Y, axis = 1)
    p = np.append(p1, p2, axis = 0)
    p = np.append(p, Z, axis = 1)
    #print(np.shape(p))
    #print(np.shape(P))
    M = np.linalg.lstsq(p, P, rcond = None)[0] 
    #result = np.dot(P,M)
    #print (result-p)
    last_row = np.zeros((4,1))
    last_row[3] = 1
    M = np.append(M, last_row, axis =1)
    
    return np.transpose(M)
    #pass
    # END YOUR CODE HERE

'''
RMS_ERROR
Arguments:
     camera_matrix - The camera matrix of the calibrated camera
     real_XY - Each row corresponds to an actual point on the 2D plane
     front_image - Each row is the pixel location in the front image where Z=0
     back_image - Each row is the pixel location in the back image where Z=150
Returns:
    rms_error - The root mean square error of reprojecting the points back
                into the images
'''
def rms_error(camera_matrix, real_XY, front_image, back_image):
    # BEGIN YOUR CODE HERE
    Z = np.zeros((12,1))
    p1 = np.append(real_XY, Z, axis = 1)
    Z = Z + 150
    p2 = np.append(real_XY, Z, axis = 1)
    p = np.append(p1, p2, axis = 0)
    Z = np.zeros((24,1)) + 1
    p = np.append(p, Z, axis = 1)
    
    Y = np.zeros((12,1))+1
    front_image = np.append(front_image, Y, axis = 1)
    back_image = np.append(back_image, Y, axis = 1)

    
    P = np.append(front_image, back_image, axis = 0)
    

    #p = np.append(p, Z, axis = 1)
    print(np.shape(camera_matrix))
    print(np.shape(P))
    print(np.shape(p))
    PM = np.matmul(p, np.transpose(camera_matrix) )
    #print(p-PM)
    Error = np.square(P - (PM)).sum()
    return(np.sqrt(Error/24))
    # END YOUR CODE HERE

'''
TEST_P2
Test function. Do not modify.
'''
def test_p2():
    # Loading the example coordinates setup
    real_XY = np.load('real_XY.npy')
    front_image = np.load('front_image.npy')
    back_image = np.load('back_image.npy')
    camera_matrix = compute_camera_matrix(real_XY, front_image, back_image)
    print("Camera Matrix:\n", camera_matrix)
    print()
    print("RMS Error: ", rms_error(camera_matrix, real_XY, front_image, back_image))


if __name__ == '__main__':
    test_p2()