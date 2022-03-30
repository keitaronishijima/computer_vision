# Project Image Filtering and Hybrid Images Stencil Code
# Based on previous and current work
# by James Hays for CSCI 1430 @ Brown and
# CS 4495/6476 @ Georgia Tech
import numpy as np
from numpy import int32, pi, exp, sqrt
from skimage import io, img_as_ubyte, img_as_float32
from skimage.transform import rescale

def my_imfilter(image, kernel):
    """
    Inputs
    - image: numpy nd-array of dim (m,n) or (m, n, c)
    - kernel: numpy nd-array of dim (k, l)
    Returns
    - filtered_image: numpy nd-array of dim of equal 2D size (m,n) or 3D size (m, n, c)
    Errors if:
    - filter/kernel has any even dimension -> raise an Exception with a suitable error message.
    """
    filtered_image = np.zeros(image.shape)
    kernel = np.rot90(np.rot90(kernel))

    #Get height and width of image
    image_height = image.shape[0]
    image_width = image.shape[1]

    # Check if the image is colored or greyscaled
    isColored = False
    if image.ndim == 3:
        isColored = True

    #Get height and width of the kernel
    kernel_h = kernel.shape[0]
    kernel_w = kernel.shape[1]

    if kernel_h == 0 or kernel_w % 2 == 0:
        raise ValueError("Dimensions of kernel must not be even")
    
    #Get number of pads required
    pad_col = (kernel_w - 1) // 2
    pad_row = (kernel_h - 1) // 2
    if isColored is True:
        image = np.pad(image, ((pad_row, pad_row),(pad_col,pad_col),(0,0)), 'reflect')
    else:
        image = np.pad(image, ((pad_row, pad_row),(pad_col,pad_col)))

    # Update the shape of resulting image
    
    # Convolve the image with greyscale
    if isColored is False:
        for i in range(pad_row, pad_row + image_height):
            for j in range(pad_col, pad_col + image_width):
                image_extract = image[i - pad_row: i + pad_row+1, j - pad_col: j + pad_col+1]
                res = np.sum(np.multiply(image_extract, kernel))
                filtered_image[i-pad_row][j-pad_col] = res

    # Convolve the image with color
    else:
        # image = image.astype(np.float)
        # print(np.sum(kernel))
        for i in range(pad_row, pad_row + image_height):
            for j in range(pad_col, pad_col + image_width):
                for color_channel in range(3):
                    image_extract = image[i - pad_row: i + pad_row+1, j - pad_col: j + pad_col+1, color_channel]
                    res = np.multiply(image_extract, kernel)
                    filtered_image[i-pad_row][j-pad_col][color_channel] = np.sum(res)
    return filtered_image


def my_imfilter_fft(image, kernel):
    """
    Inputs
    - image: numpy nd-array of dim (m,n) or (m, n, c)
    - kernel: numpy nd-array of dim (k, l)
    Returns
    - filtered_image: numpy nd-array of dim of equal 2D size (m,n) or 3D size (m, n, c)
    Errors if:
    - filter/kernel has any even dimension -> raise an Exception with a suitable error message.
    """
    filtered_image = np.zeros(image.shape)

    print('my_imfilter_fft function in student.py is not implemented')

    return filtered_image


def gen_hybrid_image(image1, image2, cutoff_frequency):
    """
     Inputs:
     - image1 -> The image from which to take the low frequencies.
     - image2 -> The image from which to take the high frequencies.
     - cutoff_frequency -> The standard deviation, in pixels, of the Gaussian
                           blur that will remove high frequencies.

     Task:
     - Use my_imfilter to create 'low_frequencies' and 'high_frequencies'.
     - Combine them to create 'hybrid_image'.
    """

    assert image1.shape[0] == image2.shape[0]
    assert image1.shape[1] == image2.shape[1]
    assert image1.shape[2] == image2.shape[2]

    # Steps:
    # (1) Remove the high frequencies from image1 by blurring it. The amount of
    #     blur that works best will vary with different image pairs
    # generate a 1x(2k+1) gaussian kernel with mean=0 and sigma = s, see https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
    s, k = cutoff_frequency, cutoff_frequency*2
    probs = np.asarray([exp(-z*z/(2*s*s))/sqrt(2*pi*s*s) for z in range(-k,k+1)], dtype=np.float32)
    kernel = np.outer(probs, probs)

    low_frequencies = my_imfilter(image1,kernel) # Replace with your implementation
    # (2) Remove the low frequencies from image2. The easiest way to do this is to
    #     subtract a blurred version of image2 from the original version of image2.
    #     This will give you an image centered at zero with negative values.

    high_frequencies = (image2 - my_imfilter(image2,kernel)) * 0.5 # Replace with your implementation
    # (3) Combine the high frequencies and low frequencies
    hybrid_image = (low_frequencies + high_frequencies)/1.5 # Replace with your implementation
    return low_frequencies, high_frequencies, hybrid_image