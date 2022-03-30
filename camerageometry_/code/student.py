import numpy as np
import cv2
import random

def calculate_projection_matrix(image, markers):
    """
    To solve for the projection matrix. You need to set up a system of
    equations using the corresponding 2D and 3D points. See the handout, Q5
    of the written questions, or the lecture slides for how to set up these
    equations.

    Don't forget to set M_34 = 1 in this system to fix the scale.

    :param image: a single image in our camera system
    :param markers: dictionary of markerID to 4x3 array containing 3D points
    :return: M, the camera projection matrix which maps 3D world coordinates
    of provided aruco markers to image coordinates
    """
    ######################
    # Do not change this #
    ######################

    # Markers is a dictionary mapping a marker ID to a 4x3 array
    # containing the 3d points for each of the 4 corners of the
    # marker in our scanning setup
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_1000)
    parameters = cv2.aruco.DetectorParameters_create()

    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(
        image, dictionary, parameters=parameters)
    markerIds = [m[0] for m in markerIds]
    markerCorners = [m[0] for m in markerCorners]

    points2d = []
    points3d = []

    for markerId, marker in zip(markerIds, markerCorners):
        if markerId in markers:
            for j, corner in enumerate(marker):
                points2d.append(corner)
                points3d.append(markers[markerId][j])

    points2d = np.array(points2d)
    points3d = np.array(points3d)

    # This M matrix came from a call to rand(3,4). It leads to a high residual.
    print('Randomly setting matrix entries as a placeholder')
    M = np.array([[0.1768, 0.7018, 0.7948, 0.4613],
                  [0.6750, 0.3152, 0.1136, 0.0480],
                  [0.1020, 0.1725, 0.7244, 0.9932]])

    num_points = points2d.shape[0]
    A_height = 2 * num_points
    A_width = 11
    A = np.zeros((A_height, A_width))
    for i in range(A_height):
        X = points3d[int(i/2)][0]
        Y = points3d[int(i/2)][1]
        Z = points3d[int(i/2)][2]
        u = points2d[int(i/2)][0]
        v = points2d[int(i/2)][1]
        if i % 2 == 0:
            A[i][0] = points3d[int(i/2)][0]
            A[i][1] = Y
            A[i][2] = Z
            A[i][3] = 1
            A[i][4] = 0
            A[i][5] = 0
            A[i][6] = 0
            A[i][7] = 0
            A[i][8] = -points3d[int(i/2)][0] * u
            A[i][9] = -Y * u
            A[i][10] = -Z * u
        elif i % 2 == 1:
            A[i][0] = 0
            A[i][1] = 0
            A[i][2] = 0
            A[i][3] = 0
            A[i][4] = points3d[int(i/2)][0]
            A[i][5] = Y
            A[i][6] = Z
            A[i][7] = 1
            A[i][8] = -points3d[int(i/2)][0] * v
            A[i][9] = -Y * v
            A[i][10] = -Z * v
    compare_uv = np.array([np.ndarray.flatten(points2d)]).T
    tmp = np.linalg.lstsq(A, compare_uv)[0]
    M_original_shape = M.shape
    M = np.zeros((12,1))
    for i in range(12):
        if i == 11:
            M[i][0] = 1
        else:
            M[i][0] = tmp[i][0]
    M = M.reshape(M_original_shape)

    return M


def normalize_coordinates(points):
    """
    ============================ EXTRA CREDIT ============================
    Normalize the given Points before computing the fundamental matrix. You
    should perform the normalization to make the mean of the points 0
    and the average magnitude 1.0.

    The transformation matrix T is the product of the scale and offset matrices

    Offset Matrix
    Find c_u and c_v and create a matrix of the form in the handout for T_offset

    Scale Matrix
    Subtract the means of the u and v coordinates, then take the reciprocal of
    their standard deviation i.e. 1 / np.std([...]). Then construct the scale
    matrix in the form provided in the handout for T_scale

    :param points: set of [n x 2] 2D points
    :return: a tuple of (normalized_points, T) where T is the [3 x 3] transformation
    matrix
    """
    
    # This is a placeholder with the identity matrix for T replace with the
    # real transformation matrix fo this set of points
    T = np.eye(3)
    T_offset = np.eye(3)
    u_mean = np.mean(points, axis = 0)[0]
    v_mean = np.mean(points, axis = 0)[1]
    T_offset[0,2] = -u_mean
    T_offset[1,2] = -v_mean
    
    T_scale = np.eye(3)
    
    points_copy = points.copy()
    points_copy[:, 0] = points_copy[:, 0] - u_mean
    points_copy[:, 1] = points_copy[:, 1] - v_mean
    s = np.std(points_copy)
    T_scale[0,0] = 1/s
    T_scale[1,1] = 1/s
    
    add = np.ones((len(points), 1))
    points = np.hstack((points, add))
    
    T = T_scale @ T_offset
    for i in range(len(points)):
        points[i] = (T @ points[i].T).T
    points = np.delete(points, 2, 1)
    return points, T


def estimate_fundamental_matrix(points1, points2):
    """
    Estimates the fundamental matrix given set of point correspondences in
    points1 and points2.

    points1 is an [n x 2] matrix of 2D coordinate of points on Image A
    points2 is an [n x 2] matrix of 2D coordinate of points on Image B

    Try to implement this function as efficiently as possible. It will be
    called repeatedly for part IV of the project

    If you normalize your coordinates for extra credit, don't forget to adjust
    your fundamental matrix so that it can operate on the original pixel
    coordinates!

    :return F_matrix, the [3 x 3] fundamental matrix
    """

    points1, Ta  = normalize_coordinates(points1)
    points2, Tb = normalize_coordinates(points2)
    F_matrix = np.array([[0, 0, -.0004], [0, 0, .0032], [0, -0.0044, .1034]])
    flag = True
    prev = None
    for i in range(len(points1)):
        u = points1[i][0]
        v = points1[i][1]
        u1 = points2[i][0]
        v1 = points2[i][1]
        r = np.array([[u * u1, v * u1, u1, u * v1, v * v1, v1, u, v, 1]])
        if flag is False:
            prev = np.vstack((prev, r))
        else:
            prev = r
            flag = False
    Q = prev
    U, S, Vh = np.linalg.svd(Q)
    F = Vh[-1,:]
    F = np.reshape(F, (3,3))
    
    U, S, Vh = np.linalg.svd(F)
    S[-1] = 0
    F = U @ np.diagflat(S) @ Vh
    F_matrix = F
    return (Tb.T) @ F_matrix @ Ta


def ransac_fundamental_matrix(matches1, matches2, num_iters):
    """
    Find the best fundamental matrix using RANSAC on potentially matching
    points. Run RANSAC for num_iters.

    matches1 and matches2 are the [N x 2] coordinates of the possibly
    matching points from two pictures. Each row is a correspondence
     (e.g. row 42 of matches1 is a point that corresponds to row 42 of matches2)

    best_Fmatrix is the [3 x 3] fundamental matrix, inliers1 and inliers2 are
    the [M x 2] corresponding points (some subset of matches1 and matches2) that
    are inliners with respect to best_Fmatrix

    For this section, use RANSAC to find the best fundamental matrix by randomly
    sampling interest points. You would call the function that estimates the 
    fundamental matrix (either the "cheat" function or your own 
    estimate_fundamental_matrix) iteratively within this function.

    If you are trying to produce an uncluttered visualization of epipolar lines,
    you may want to return no more than 30 points for either image.

    :return: best_Fmatrix, inliers1, inliers2
    """
    # DO NOT TOUCH THE FOLLOWING LINES
    random.seed(0)
    np.random.seed(0)

    best_Fmatrix = estimate_fundamental_matrix(matches1[0:9, :], matches2[0:9, :])
    inliers_a = np.zeros((0,0))
    inliers_b = np.zeros((0,0))
    num_samples = len(matches1)
    thr = 0.1
    best_count = 0
    for i in range(5000):
        rand_indices= list(range(0,matches1.shape[0]))
        random.shuffle(rand_indices)
        test1 = np.zeros((9,2))
        test2 = np.zeros((9,2))
        for j in range(9):
            test1[j] = matches1[rand_indices[j]]
            test2[j] = matches2[rand_indices[j]]
        tmp = estimate_fundamental_matrix(test1, test2)
        in_count = 0
        inliers_a_tmp = None
        inliers_b_tmp = None
        ini = True
        for j in range(num_samples):
            x = np.array([np.append(matches1[j], 1)])
            x_dash = np.array([np.append(matches2[j], 1)])
            error = x @ tmp @ x_dash.T
            if abs(error) < thr:
                if ini is True:
                    inliers_a_tmp = np.array([matches1[j]])
                    inliers_b_tmp = np.array([matches2[j]])
                    ini = False
                else:
                    inliers_a_tmp = np.hstack((inliers_a_tmp,np.array([matches1[j]])))
                    inliers_b_tmp = np.hstack((inliers_b_tmp, np.array([matches2[j]])))
                in_count += 1
        if in_count > best_count:
            best_Fmatrix = tmp
            inliers_a = inliers_a_tmp
            inliers_b = inliers_b_tmp
            best_count = in_count
    inliers_a = inliers_a.reshape((len(inliers_a[0])//2, 2))
    inliers_b = inliers_b.reshape((len(inliers_b[0])//2, 2))

    return best_Fmatrix, inliers_a, inliers_b


def matches_to_3d(points1, points2, M1, M2):
    """
    Given two sets of points and two projection matrices, you will need to solve
    for the ground-truth 3D points using np.linalg.lstsq(). For a brief reminder
    of how to do this, please refer to Question 5 from the written questions for
    this project.


    :param points1: [N x 2] points from image1
    :param points2: [N x 2] points from image2
    :param M1: [3 x 4] projection matrix of image1
    :param M2: [3 x 4] projection matrix of image2
    :return: [N x 3] list of solved ground truth 3D points for each pair of 2D
    points from points1 and points2
    """
    points3d = []

    for i in range(len(points1)):
        u1 = points1[i][0]
        v1 = points1[i][1]
        u2 = points2[i][0]
        v2 = points2[i][1]
        first_row = M1[2, :3] * u1 - M1[0, :3]
        second_row = M1[2, :3] * v1 - M1[1, :3]
        third_row = M2[2, :3] * u2 - M2[0, :3]
        forth_row = M2[2, :3] * v2 - M2[1, :3]
        Q = np.vstack((first_row, second_row,third_row,forth_row))
        y = np.zeros((4,1))
        y[0,0] = M1[0][3] - M1[2][3] * u1
        y[1,0] = M1[1][3] - M1[2][3] * v1
        y[2,0] = M2[0][3] - M2[2][3] * u2
        y[3,0] = M2[1][3] - M2[2][3] * v2
        point3d = np.linalg.lstsq(Q, y)[0].reshape((3))
        points3d.append(point3d)
    points3d = np.array(points3d)
    return np.ndarray.tolist(points3d)
