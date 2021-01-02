# Audio Declipping 


This repositorie implement in Python the algorithms proposed in "Sparsity and Cosparsity for audio declipping: a flexible non convex approach".


A similar implementation in MATLAB is available at:
[[http://www.utko.feec.vutbr.cz/~rajmic/software/aspade_vs_sspade.zip][http://www.utko.feec.vutbr.cz/~rajmic/software/aspade_vs_sspade.zip]] 

This Python version has been designed to be easily extensible, but may lack some optimisation tweaks. 

# Description of the python implementation 
  *Packages`numpy` ,`scipy` , `matplotlib,` and `tqdm` (for progress bar ) are required.*
  
  - `declipper.py` provides a class gathering all the main steps of the problem, (simulation, reconstruction, metrics analysis)
  - `gamma.py`  implement the projection on a convex set, define by its boundarys (see report)
  - `stft.py` provides a nice porcelaine to apply Short Time Fourier Transform, as a frame operator.
  
  A basic usage is avaible in `audioDeclipping.py`. 
  
  
 
 
