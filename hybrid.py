import sys
sys.path.append('/Users/kb/bin/opencv-3.1.0/build/lib/')
import cv2
import numpy as np

def cross_correlation_2d(img, kernel):
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    m = kernel.shape[0]
    n = kernel.shape[1]

    if len(img.shape) == 2:
        height = img.shape[0]
        width  = img.shape[1]
        ret = np.zeros((height,width))
        
        for i in range(0, height):
            for j in range(0, width):
                z = 0.0
                for x in range(0, m):
                    for y in range(0, n):
                        if (i - m / 2 + x) >= 0 and (i - m / 2 + x) < height and (j - n / 2 + y) >= 0 and (j - n / 2 + y) < width:
                            z += img[i - m / 2 + x, j - n / 2 + y] * kernel[x, y]
                ret[i, j] = z
        return ret

    if len(img.shape) == 3:
        height = img.shape[0]
        width  = img.shape[1]
        ret = np.zeros((height,width,3))
 
        for i in range(0, height):
            for j in range(0, width):
                for c in range(0, 3):
                    z = 0.0
                    for x in range(0, m):
                        for y in range(0, n):
                            if (i - m / 2 + x) >= 0 and (i - m / 2 + x) < height and (j - n / 2 + y) >= 0 and (j - n / 2 + y) < width:
                                z += img[i - m / 2 + x, j - n / 2 + y, c] * kernel[x, y]
                    ret[i, j, c] = z
        return ret
    
    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def convolve_2d(img, kernel):
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN:
    m = kernel.shape[0]
    n = kernel.shape[1]
    t_ker = np.zeros((m, n))
    for i in range(0, m):
        for j in range(0, n):
            t_ker[i, j] = kernel[m - i - 1, n - j - 1]
    return cross_correlation_2d(img, t_ker) 
    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def gaussian_blur_kernel_2d(sigma, width, height):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions width x height such that convolving it
        with an image results in a Gaussian-blurred image.
    '''
    # TODO-BLOCK-BEGIN
    
    kernel = np.zeros(shape=(width,height))

    for i in range (0, height):
        for j in range (0, width):
            x = i-height/2
            y = j-width/2
            kernel[j,i] = np.exp(-((float(x)**2+y**2)/(2*sigma**2)))
    
    kernel = kernel / np.sum(kernel)

    return kernel
    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    kernel = gaussian_blur_kernel_2d(sigma, size, size)
    return convolve_2d(img, kernel)
    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    return img - low_pass(img, sigma, size)
    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *= 2 * (1 - mixin_ratio)
    img2 *= 2 * mixin_ratio
    hybrid_img = (img1 + img2)
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)


