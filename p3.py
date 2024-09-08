# CS231A Homework 1, Problem 3
import numpy as np
from utils import mat2euler
import math

'''
COMPUTE_VANISHING_POINTS
Arguments:
    points - a list of all the points where each row is (x, y). 
            It will contain four points: two for each parallel line.
Returns:
    vanishing_point - the pixel location of the vanishing point
'''
def compute_vanishing_point(points):
    # BEGIN YOUR CODE HERE
    l1 = points[0,:]-points[1,:]
    l2 = points[2,:]-points[3,:]
    M = [[-l1[1],l1[0]],[-l2[1],l2[0]]]
    c =[-l1[1]*points[0,0]+l1[0]*points[0,1],-l2[1]*points[2,0]+l2[0]*points[2,1]]
    M_inv = np.linalg.inv(M)  
    X = np.matmul(M_inv,c)
    return X    
    # END YOUR CODE HERE

'''
COMPUTE_K_FROM_VANISHING_POINTS
Makes sure to make it so the bottom right element of K is 1 at the end.
Arguments:
    vanishing_points - a list of vanishing points

Returns:
    K - the intrinsic camera matrix (3x3 matrix)
'''
def compute_K_from_vanishing_points(vanishing_points):
    # BEGIN YOUR CODE HERE
    A = []
    for i, pt_i in enumerate(vanishing_points):
        for j, pt_j in enumerate(vanishing_points):
            if i != j and j > i:
                print(pt_i)
                pt_i_homogeneous = [pt_i[0],pt_i[1],1.0]
                pt_j_homogeneous = [pt_j[0],pt_j[1],1.0]
                A.append([pt_i_homogeneous[0]*pt_j_homogeneous[0]+pt_i_homogeneous[1]*pt_j_homogeneous[1], \
                          pt_i_homogeneous[0]*pt_j_homogeneous[2]+pt_i_homogeneous[2]*pt_j_homogeneous[0], \
                          pt_i_homogeneous[1]*pt_j_homogeneous[2]+pt_i_homogeneous[2]*pt_j_homogeneous[1], \
                          pt_i_homogeneous[2]*pt_j_homogeneous[2]])
    A = np.array(A, dtype=np.float32)
    u, s, v_t = np.linalg.svd(A, full_matrices=True)

    w1, w4, w5, w6 = v_t.T[:,-1]

    w = np.array([[w1, 0., w4],
                  [0., w1, w5],
                  [w4, w5, w6]])

    K_transpose_inv = np.linalg.cholesky(w)
    K = np.linalg.inv(K_transpose_inv.T)

    K = K / K[-1, -1]
    # END YOUR CODE HERE
    return K

'''
COMPUTE_ANGLE_BETWEEN_PLANES
Arguments:
    vanishing_pair1 - a list of a pair of vanishing points computed from lines within the same plane
    vanishing_pair2 - a list of another pair of vanishing points from a different plane than vanishing_pair1
    K - the camera matrix used to take both images

Returns:
    angle - the angle in degrees between the planes which the vanishing point pair comes from2
'''
def compute_angle_between_planes(vanishing_pair1, vanishing_pair2, K):
    # BEGIN YOUR CODE HERE

    a = vanishing_pair1[0][1] - vanishing_pair1[1][1]
    b = vanishing_pair1[0][0] - vanishing_pair1[1][0]
    c = b*vanishing_pair1[1][1] -a*vanishing_pair1[1][0]
    vanishing_line1 = np.asarray([a,-b,c])

    a = vanishing_pair2[0][1] - vanishing_pair2[1][1]
    b = vanishing_pair2[0][0] - vanishing_pair2[1][0]
    c = b*vanishing_pair2[1][1] -a*vanishing_pair2[1][0]
    vanishing_line2 = np.asarray([a,-b,c])
    

    w_inv = np.dot(K, K.transpose())

    l1T_winv_l2 = np.dot(vanishing_line1.transpose(), np.dot(w_inv, vanishing_line2))
    sqrt_l1T_winv_l1 = np.sqrt(np.dot(vanishing_line1.transpose(), np.dot(w_inv, vanishing_line1)))
    sqrt_l2T_winv_l2 = np.sqrt(np.dot(vanishing_line2.transpose(), np.dot(w_inv, vanishing_line2)))
    theta = np.arccos(l1T_winv_l2 / np.dot(sqrt_l1T_winv_l1, sqrt_l2T_winv_l2))

    return np.degrees(theta)

    # END YOUR CODE HERE

'''
COMPUTE_ROTATION_MATRIX_BETWEEN_CAMERAS
Arguments:
    vanishing_points1 - a list of vanishing points in image 1
    vanishing_points2 - a list of vanishing points in image 2
    K - the camera matrix used to take both images

Returns:
    R - the rotation matrix between camera 1 and camera 2
'''
def compute_rotation_matrix_between_cameras(vanishing_points1, vanishing_points2, K):
    # BEGIN YOUR CODE HERE
    d1i = []
    for v1i in vanishing_points1:
 
        v1i_homogeneous = np.array([v1i[0], v1i[1], 1.0])
        KinvV = np.dot(np.linalg.inv(K), v1i_homogeneous.T)
        d1i.append(KinvV / np.sqrt(KinvV[0]**2 + KinvV[1]**2 + KinvV[2]**2)) 
    d1i = np.array(d1i)

    d2i = []
    for v2i in vanishing_points2:
 
        v2i_homogeneous = np.array([v2i[0], v2i[1], 1.0])
        KinvV = np.dot(np.linalg.inv(K), v2i_homogeneous.T)
        d2i.append(KinvV / np.sqrt(KinvV[0]**2 + KinvV[1]**2 + KinvV[2]**2)) 
    d2i = np.array(d2i)
    

    R = np.dot(d2i.T, np.linalg.inv(d1i.T))
    return R
    # END YOUR CODE HERE

'''
TEST_P3
Test function. Do not modify.
'''
def test_p3():
    # Part A: Compute vanishing points
    v1 = compute_vanishing_point(np.array([[1080, 598],[1840, 478],[1094,1340],[1774,1086]]))
    v2 = compute_vanishing_point(np.array([[674,1826],[4, 878],[2456,1060],[1940,866]]))
    v3 = compute_vanishing_point(np.array([[1094,1340],[1080,598],[1774,1086],[1840,478]]))

    v1b = compute_vanishing_point(np.array([[314,1912],[2060,1040],[750,1378],[1438,1094]]))
    v2b = compute_vanishing_point(np.array([[314,1912],[36,1578],[2060,1040],[1598,882]]))
    v3b = compute_vanishing_point(np.array([[750,1378],[714,614],[1438,1094],[1474,494]]))

    # Part B: Compute the camera matrix
    vanishing_points = [v1, v2, v3]
    print("Intrinsic Matrix:\n",compute_K_from_vanishing_points(vanishing_points))

    K_actual = np.array([[2448.0, 0, 1253.0],[0, 2438.0, 986.0],[0,0,1.0]])
    print()
    print("Actual Matrix:\n", K_actual)

    # Part D: Estimate the angle between the box and floor
    floor_vanishing1 = v1
    floor_vanishing2 = v2
    box_vanishing1 = v3
    box_vanishing2 = compute_vanishing_point(np.array([[1094,1340],[1774,1086],[1080,598],[1840,478]]))
    angle = compute_angle_between_planes([floor_vanishing1, floor_vanishing2], [box_vanishing1, box_vanishing2], K_actual)
    print()
    print("Angle between floor and box:", angle)

    # Part E: Compute the rotation matrix between the two cameras
    rotation_matrix = compute_rotation_matrix_between_cameras(np.array([v1, v2, v3]), np.array([v1b, v2b, v3b]), K_actual)
    print("Rotation between two cameras:\n", rotation_matrix)
    z,y,x = mat2euler(rotation_matrix)
    print()
    print("Angle around z-axis (pointing out of camera): %f degrees" % (z * 180 / math.pi))
    print("Angle around y-axis (pointing vertically): %f degrees" % (y * 180 / math.pi))
    print("Angle around x-axis (pointing horizontally): %f degrees" % (x * 180 / math.pi))


if __name__ == '__main__':
    test_p3()
