# cells-in-gel
## Overall project description: 
This is the repo of team Cells-in-gels. Most of us work in the DeForest group, studying how biomaterials can influence cell behavior. As such, this project analyzes how functionalizing poly(ethylene glycol) hydrogels with the adhesion motif RGD in different concentrations affects cell spreading and proliferation of fibroblasts. We will design a pipeline for processing fluorescent images taken on a Leica confocal. 


## Installation
In order to run cells-in-gel, clone the repository onto your local computer. Packages required by this project are: matplotlib, scikit-image, and numpy. 

## Overview of scripts and functions:
We import z-stacks of a hydrogel from three different channels from the Leica confocal. 
- Channel 1 = DAPI, stains nuclei. 
- Channel 2 = actin, stains actin filaments of cells. 
- Channel 3 = aSMA, stains alpha-smooth muscle actin which is a protein expressed in activated fibroblasts. 

However, channel 3, which is used to measure the protein expression level can be adapted for other proteins. 

For channel 1, we write a script and functions to sharpen and isolate the nuclei from the background to count the number of cells. For channel 2, we set a threshold to separate the actin filaments (which basically outline the cytoplasm of the nucleus) from the background, binarize the image, and then calculate the area and extension of the cells. For channel 3, we quantify the intensity of the aSMA, which can then be normalized against the number of cells calculated in channel 1.

## Use cases:
This program is designed for evaluating the effects of a 3D environment on cells' spreading and expression of proteins. The use of channel 3 can be adapted. For instance, one could measure extracellular matrix protein (i.e. collagen, laminin, fibronectin) deposition intensity, or the expression of Postn (another fibroblast activation marker). 
![Fibroblasts stained for Collagen I](0.5mM_BMB_maxproj_01.tif)
