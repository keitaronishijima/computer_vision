from re import X
import matplotlib
import skimage
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab
import scipy
from skimage import filters, feature, img_as_int
from skimage.measure import regionprops



def plot_interest_points(image, x, y):
    '''
    Plot interest points for the input image. 
    
    Show the interest points given on the input image. Be sure to add the images you make to your writeup. 
    Useful functions: Some helpful (not necessarily required) functions may include
        - matplotlib.pyplot.imshow, matplotlib.pyplot.scatter, matplotlib.pyplot.show, matplotlib.pyplot.savefig
    
    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates of interest points
    :y: np array of y coordinates of interest points
    '''

    matplotlib.pyplot.scatter(x,y)
    plt.imshow(image)
    plt.show()


def get_interest_points(image, feature_width):
    '''
    Returns interest points for the input image
    (Please note that we recommend implementing this function last and using cheat_interest_points()
    to test your implementation of get_features() and match_features())
    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You do not need to worry about scale invariance or keypoint orientation estimation
    for your Harris corner detector.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.
    If you're finding spurious (false/fake) interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.
    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions
        - skimage.feature.peak_local_max (experiment with different min_distance values to get good results)
        - skimage.measure.regionprops
          
    Note: You may decide it is unnecessary to use feature_width in get_interest_points, or you may also decide to 
    use this parameter to exclude the points near image edges.
    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :feature_width: the width and height of each local feature in pixels
    :returns:
    :xs: an np array of the x coordinates of the interest points in the image
    :ys: an np array of the y coordinates of the interest points in the image
    :optional returns (may be useful for extra credit portions):
    :confidences: an np array indicating the confidence (strength) of each interest point
    :scale: an np array indicating the scale of each interest point
    :orientation: an np array indicating the orientation of each interest point
    '''

    # These are placeholders - replace with the coordinates of your interest points!

    xs = np.zeros(1)
    ys = np.zeros(1)
    # Note that xs and ys represent the coordinates of the image. Thus, xs actually denote the columns
    # of the respective points and ys denote the rows of the respective points.

    # STEP 1: Calculate the gradient (partial derivatives on two directions).
    gradients_y, gradients_x = np.gradient(image)
    # STEP 2: Apply Gaussian filter with appropriate sigma.
    # STEP 3: Calculate Harris cornerness score for all pixels.
    Ixx = scipy.ndimage.gaussian_filter(gradients_x**2, sigma=2.5)
    Ixy = scipy.ndimage.gaussian_filter(gradients_y*gradients_x, sigma=2.5)
    Iyy = scipy.ndimage.gaussian_filter(gradients_y**2, sigma=2.5)

    k = 0.05

    # determinant
    detA = Ixx * Iyy - Ixy ** 2
    # trace
    traceA = Ixx + Iyy
        
    harris_cornerness = (detA - k * (traceA * traceA)) * 10000000

    # STEP 4: Peak local max to eliminate clusters. (Try different parameters.)
    cornerness_sum = np.sum(harris_cornerness)
    if cornerness_sum < 0:
        cornerness_sum = cornerness_sum * (-1) + cornerness_sum + 100000
    coords = skimage.feature.peak_local_max(harris_cornerness, min_distance=3, num_peaks = 1950)

    for i in range(len(coords)):
        ys = np.append(ys,coords[i,0])
        xs = np.append(xs,coords[i,1])
    return xs, ys


def get_features(image, x, y, feature_width):
    '''
    Returns features for a given set of interest points.
    To start with, you might want to simply use normalized patches as your
    local feature descriptor. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT-like feature descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)
    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) feature descriptor should have:
    (1) a 4x4 grid of cells, each feature_width / 4 pixels square.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length
    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.
    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like features can be constructed entirely
    from filtering fairly quickly in this way.
    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.
    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.
    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions
        - skimage.filters (library)
    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates of interest points
    :y: np array of y coordinates of interest points
    :feature_width: in pixels, is the local feature width. You can assume
                    that feature_width will be a multiple of 4 (i.e. every cell of your
                    local SIFT-like feature will have an integer width and height).
    If you want to detect and describe features at multiple scales or
    particular orientations you can add input arguments. Make sure input arguments 
    are optional or the autograder will break.
    :returns:
    :features: np array of computed features. It should be of size
            [len(x) * feature dimensionality] (for standard SIFT feature
            dimensionality is 128)
    '''

    feature_width = 68
    # STEP 1: Calculate the gradient (partial derivatives on two directions) on all pixels.

    gradients_x = filters.sobel(image, axis = 1)
    gradients_y = filters.sobel(image, axis = 0)

    gradients_y, gradients_x = np.gradient(image)

    cell = feature_width // 4
    # STEP 2: Decompose the gradient vectors to magnitude and direction.

    gradient_orientations = np.arctan2(gradients_x, gradients_y)
    gradient_magnitudes = np.sqrt(np.square(gradients_x) + np.square(gradients_y))

    for i in range(len(gradient_orientations)):
        for j in range(len(gradient_orientations[i])):
            if gradient_orientations[i,j] < 0:
                gradient_orientations[i,j] = 2 * np.pi + gradient_orientations[i,j]

    desc = 32
    s = desc * desc * 16
    features = np.zeros((len(x), s))
    # STEP 3: For each interest point, calculate the local histogram based on related 4x4 grid cells.
    #         Each cell is a square with feature_width / 4 pixels length of side.
    #         For each cell, we assign these gradient vectors corresponding to these pixels to 8 bins
    #         based on the direction (angle) of the gradient vectors. 

    # STEP 4: Now for each cell, we have a 8-dimensional vector. Appending the vectors in the 4x4 cells,
    #         we have a 128-dimensional feature.
    
    f_l = len(features)
    for m in range(f_l):
        col = int(x[m])
        row = int(y[m])
        for i in range(4):
            for j in range(4):
                descriptor_index_head = i * cell * desc + j * desc
                lower_k = row - 2 * cell + i * cell
                upper_k = row - 2*cell + (i + 1) * cell
                for k in range(lower_k, upper_k):
                    lower_l = col - 2*cell + j * cell
                    upper_l = col - 2*cell + (j + 1) * cell
                    for l in range(lower_l, upper_l):
                        if k >= gradient_orientations.shape[0]-1 or l >= gradient_orientations.shape[1]-1:
                            continue
                        else:
                            offset = int(gradient_orientations[k,l] // (2 * np.pi/desc))
                            features[m, descriptor_index_head + offset] += gradient_magnitudes[k,l]

    # STEP 5: Don't forget to normalize your feature.
    sum_of_rows = features.sum(axis=1)
    features = features / sum_of_rows[:, np.newaxis]

    return features


def match_features(im1_features, im2_features):
    '''
    Implements the Nearest Neighbor Distance Ratio Test to assign matches between interest points
    in two images.
    Please implement the "Nearest Neighbor Distance Ratio (NNDR) Test" ,
    Equation 4.18 in Section 4.1.3 of Szeliski.
    For extra credit you can implement spatial verification of matches.
    Please assign a confidence, else the evaluation function will not work. Remember that
    the NNDR test will return a number close to 1 for feature points with similar distances.
    Think about how confidence relates to NNDR.
    This function does not need to be symmetric (e.g., it can produce
    different numbers of matches depending on the order of the arguments).
    A match is between a feature in im1_features and a feature in im2_features. We can
    represent this match as a the index of the feature in im1_features and the index
    of the feature in im2_features
    :params:
    :im1_features: an np array of features returned from get_features() for interest points in image1
    :im2_features: an np array of features returned from get_features() for interest points in image2
    :returns:
    :matches: an np array of dimension k x 2 where k is the number of matches. The first
            column is an index into im1_features and the second column is an index into im2_features
    :confidences: an np array with a real valued confidence for each match
    '''
        
    # STEP 1: Calculate the distances between each pairs of features between im1_features and im2_features.

    f1_squared = np.array([np.sum(np.square(im1_features), axis = 1)]).T # should be n by 1
    f2_squared_t = np.array([np.sum(np.square(im2_features), axis = 1)]) # should be 1 by m
    tmp = f1_squared + f2_squared_t
    two_f1_f2 = 2 * (np.dot(im1_features, im2_features.T)) # should be n by m
    dist_list = np.sqrt(tmp - two_f1_f2)  # should be n ny m
    # STEP 2: Sort and find closest features for each feature, then performs NNDR test.

    matches = np.zeros((im1_features.shape[0],2)) # should be n by 2
    confidences = np.zeros(im1_features.shape[0])
   
    for i in range(len(dist_list)):
        max_val_index = np.argmin(dist_list[i])
        tmp = np.sort(dist_list[i])
        # Don't calculate confidence if there is only one feature!!
        if dist_list.shape[1] != 1:
            d1 = tmp[0]
            d2 = tmp[1]
            confidences[i] = 1 - d1/d2
        matches[i][0] = i
        matches[i][1] = max_val_index
    return matches, confidences
