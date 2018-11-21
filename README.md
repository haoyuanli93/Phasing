# Phasing
This package is an implementation of the HIO algorithm and the 
shrinkwrap function described in the papaer

S. Marchesini, H. He, H. N. Chapman, S. P. Hau-Riege, 
A. Noy, M. R. Howells, U. Weierstall, and J. C. H. Spence
Phys. Rev. B 68, 140101(R) – Published 28 October 2003

Originally the code is borrowed from the repo:

https://github.com/cwg45/Image-Reconstruction

The algorithm implemented in that repo is from the following 
paper. 

J. R. Fienup, "Phase retrieval algorithms: a
 comparison," Appl. Opt. 21, 2758-2769 (1982)

Now I have completely rewritten the code. Therefore no part
of this repo should come from the above repo. However, 
I am not sure if there are still
some remnant codes are from that repo. Therefore I'll still 
keep the link here for record. I'll fix it as soon as possible 
if anyone claim the copyright of any part of this repo.

## Introduction
This repo, as is depicted in the above section, is an 
implementation of the HIO algorithm with the shrinkwrap
function. Why another repo on this? 

First, I am a PhD student and I have some ideas to try with 
this algorithm. The existing implementations are not very
convenient for this purpose. 

Second, I'd like to have a package working with gpu and is 
convenient to use inside the jupyter notebook. 

So, here I have developed this repo. I have finished the first 
stage. i.e. It can do HIO with shrinkwrap on both cpu and gpu
now. However, it's not published to pip yet. Therefore, 
if one would like to use this repo, one has to clone this 
repo and add the path as I have done in the examples in the 
'notebooks' folder.

So much intro for now. I'll temporarily move to the other 
projects now. Hopefully, I'll come back to this soon and 
finish the next step. 