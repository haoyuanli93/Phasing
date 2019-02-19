# Phasing

This project aims to provide convenient commandline toolkit for scientists to solve phase 
retrieval problems with alternating projection algorithms such as ER, HIO, HPR, RAAR and GIF-RAAR.


At present, this algorithm support the following algorithms

- Error Reduction (ER)
- Hybrid Input-Output [(HIO)](https://www.osapublishing.org/ol/abstract.cfm?uri=ol-3-1-27)
- Hybrid Projection Reflection [(HPR)](https://www.ncbi.nlm.nih.gov/pubmed/12801170)
- Relaxed Averaged Alternating Reflections [(RAAR)](https://iopscience.iop.org/article/10.1088/0266-5611/21/1/004)
- Relaxed Averaged Alternating Reflections with Generalized Interior Feedback [(GIF-RAAR)](https://www.ncbi.nlm.nih.gov/pubmed/23187243) 
- [Shrink-Wrap](https://link.aps.org/doi/10.1103/PhysRevB.68.140101) 

In this package, sections begin with '[Developer]' means that this section is written for the next developer, though user with some python experience are also encouraged to read that section to understand how this package works better.

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

# [Developer] Project Structure

This package is divided in to two layers. 

The first layer is in CpuUtil.py. At present, it contains all the alternating projections operators. After studying these algorithms, one can realize that many of the existing alternating projections algorithms can be written in a similar format. The only difference is the relation between the coefficients. Therefore, the functions in this module is rather general. It does the general alternating projection operations.

The reason to design it in this way, is that, in practice, I have never applied the operator for more than 10^4 times. Therefore, I think it will not slow down the calculation to put the function in another module.

The second layer is in AlterProj.py. In this function, there is an object controlling the actual calculation process. In practice, I imagine the user will create one such object, then use the object's methods to hold the data, derive support information, tune shrinkwrap behavior and tune the algorithm behavior. 

The reason to design such an object is multiple. Because the majority operations in the function is linear or simple elementwise operations, it's very straightforward to implement them with numpy array and numba functions. However, there will be quite a lot of auxiliary variables to finish all the calculation. To reduce the time consumed by memory allocation, I will create all the auxiliary variables required in this AlterPorj object. Then send them to the operation functions through dictionary to reduce the memory allocation. Also, in the operation functions, I have pay attention to reduce the memory allocation as much as I can.

Previously, there is another layer in the ToolChain.py file. However, when I tune this package, I found it a little bit clumsy. Therefore,it's abandoned now. However, it does not mean it's totally worthless. In the future, if one would like to tune its default behavior carefully and find several very robust solutions to several general and common siturations, this object can still be useful. At present, the ToolChain.py module is stored in the folder OutdateFiles.

# Example and working flow
Examples are stored in jupyer notebook files in the *Example* folder. 

There are several things to notice. 

1. Usually, one needs to do multiple reconstructions and compare all the reconstructed densities with correlatoins and selected the best one. This is best done with MPI. However, I did not implement that. Instead, one has to write such a script oneself. 
2. Theoretically, the difference between 2D and 3D is mainly about the speed. However, in reality, the different is huge. 2D patterns are mainly photon counts which suffer from serious Poisson noise while 3D patterns are mainly reconstructed smooth patterns. No Poisson noise, but there are some other more subtle noise.
3. The detector gaps and beam stops could have significant influences on the initial support which in turn can greatly influence the reconstructed image.
4. I have not implemented any metrics to measure the convergence.
5. The function *resolve_trivial_ambiguity* in the util.py module can shift the image to the center and search all possible flips to resolve the trivial ambiguity.
6. I find the combination of RAAR and shrink-wrap not so stable. For me, the best practice seems to be 

    1. Use the autocorrelation support and RAAR to obtain a general solution.
    2. Use shrink-wrap + ER to reduce the support.
    3. Use RAAR with the new support to obtain an approximated solution.
    4. Use ER with the existing support to reduce errors.

In the example folder, the following situaitons are tested.

1. No noise, no gaps, support from auto-correlation
2. No noise, no gaps, improved support with shrink-wrap
3. Noisy, with beam stop, support from auto-correlation
4. Noisy, with beam stop, improved support with shrink-wrap

# Tune the behavior
This package is quite flexible. One can tune it's behavior with different initialization methods.

## Simple initialization

1. Create a AlterProj object. This object is the interface for all calculations.
2. Initialize the object with the magnitude and the mask for the magnitude. Notice that the 
magnitude should have been shifted with numpy.fft.ifftshift
3. Set the behavior of the object to be the default behavior
4. Execute the algorithm

## Complex initialization

1. Create a AlterProj object. This object is the interface for all the later on calculation.
2. Initialize the object with the magnitude and the mask for the magnitude. Notice that the 
magnitude should have been shifted with numpy.fft.ifftshift
3. Initialize the support with whatever you think is proper.
3. Initialize the algorithm 
4. Set the behavior of the shrink-wrap algorithm.

# Package Dependence
This repo depends on the following packages:

    1.numpy
    2.numba
    5.scipy
    6.scikit-image
 
To run the examples, one needs additionally the following
packages:

    1. matplotlib
    2. jupyter notebook

# Important Record
This section contains some important record for later development.

## What to be done
- Add a more realistic example

## What can be done for the next developer 
- Add GPU support. Therefore, if the user needs to analysis a large object, then the GPU can 
speed up the calculation a lot. 
- Add metrics to measure the convergence progress and the convergence property.
- Add mpi4py examples.