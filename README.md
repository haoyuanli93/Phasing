# Phasing

This package aims to give users a good control of the process when they want to 
retrieve the phase of diffraction patterns with various alternating projection 
algorithms.

At present, this algorithm support the following algorithms

- Error Reduction (ER)
- Hybrid Input-Output [(HIO)](https://www.osapublishing.org/ol/abstract.cfm?uri=ol-3-1-27)
- Hybrid Projection Reflection [(HPR)](https://www.ncbi.nlm.nih.gov/pubmed/12801170)
- Relaxed Averaged Alternating Reflections [(RAAR)](https://iopscience.iop.org/article/10.1088/0266-5611/21/1/004)
- Relaxed Averaged Alternating Reflections with Generalized Interior Feedback [(GIF-RAAR)](https://www.ncbi.nlm.nih.gov/pubmed/23187243) 
- [Shrink-Wrap](https://link.aps.org/doi/10.1103/PhysRevB.68.140101) 

#### Reference 
In the previous section, links are connected to papers for the corresponding algorithm. However, 
I have not been able to produce a complete references because I am not familiar with markdown. 
I'll come back and add a complete reference in about 2 months (i.e. 2019.3).

This package grew from the following repo: [https://github.com/cwg45/Image-Reconstruction](https://github.com/cwg45/Image-Reconstruction)
It only implements the HIO algorithm. I begin this project when I was playing with the script in 
this repo. 

I think I have completely rewritten the code, especially in this second version. However, 
I am not sure if there are still some remnant codes coming from that repo. Therefore I'll still 
keep the link here for record. 

## Project Goal
This project aims to provide convenient commandline toolkit for scientists to solve phase 
retrieval problems with alternating projection algorithms such as ER, HIO, HPR, RAAR and GIF-RAAR.

## Package Dependence
This repo depends on the following packages:

    1.numpy
    2.numba
    3.cudatoolkit
    4.pyculib
    5.scipy
    6. scikit-image
 
To run the examples, one needs additionally the following
packages:

    1. matplotlib
    2. jupyter notebook
    
## Detailed Record
Honestly, I have not idea how to name this section. There is quite a lot I would like to 
elaborate here.

### What has been done
First, the user can use CPU for simulation with the available algorithms now. However, there are 
still a lot to be done.

### What to be done
- Add MPI support. Therefore, later, the user can run different retrieval in parallel.
- Add GPU support. Therefore, if the user needs to analysis a large object, then the GPU can 
speed up the calculation a lot. 
- However, I have no plan for multiple GPU reconstruction in parallel. Because it's just a 
little bit too tricky at present.
- Add metrics to measure the convergence progress and the convergence property.
- Add some simple function such at the shifting to the center to make the package easier to use.

### What's the working flow with this package

#### Simple initialization

1. Create a AlterProj object. This object is the interface for all the later on calculation.
2. Initialize the object with the magnitude and the mask for the magnitude. Notice that the 
magnitude should have been shifted with numpy.fft.ifftshift
3. Set the behavior of the object to be the default behavior
4. Execute the algorithm

#### Complex initialization

1. Create a AlterProj object. This object is the interface for all the later on calculation.
2. Initialize the object with the magnitude and the mask for the magnitude. Notice that the 
magnitude should have been shifted with numpy.fft.ifftshift
3. Initialize the support with whatever you think is proper.
3. Initialize the algorithm 
4. Set the behavior of the shrink-wrap algorithm.



