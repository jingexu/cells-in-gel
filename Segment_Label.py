from skimage.filters import threshold_triangle
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.morphology import closing, square
from skimage.color import label2rgb
import numpy as np
import matplotlib.pyplot as plt

def colorize(image, i):
    """
    Signature: colorize(*args)
    Docstring: segment and label image
    Extended Summary:
    ---------------- 
    The colorize function defines the threshold value for the desired image by the triangle function and then creates a binarized image by setting pixel intensities above that thresh value to white, and the ones below to black (background). Next, it closes up the image by filling in random noise within the cell outlines and smooths/clears out the border. It then labels adjacent pixels with the same value and defines them as a region. It returns an RGB image with color-coded labels.
    
    Parameters:
    ----------
    image : 2D array
            greyscale image
    i : int
        dimension of square to be used for binarization
        
    Returns:
    --------
    RGB image
    int : 3D ndarray
          shape of image
    """
    #applying threshold to image
    thresh = threshold_triangle(image)
    binary = closing(image > thresh, square(i))
    
    #cleaning up boundaries of cells
    cleared = clear_border(binary)
    
    #labelling regions that are cells
    label_image = label(cleared)
    
    #coloring labels over cells
    image_label_overlay = label2rgb(label_image, image=image, bg_label=0)
    print(image_label_overlay.shape)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image_label_overlay)

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
