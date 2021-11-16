# Dual_Focal_Loss
Dual Focal Loss (DFL) function for neural networks.

We proposed a new loss function: Dual Focal Loss, for neural networks in our recently published paper (https://doi.org/10.1016/j.neucom.2021.07.055). This loss function was inspired by the recently proposed Focal Loss (https://doi.org/10.1109/ICCV.2017.324) and Dual Cross Entropy (https://doi.org/10.1109/TVT.2019.2895651) functions.

The MATLAB file: dfl_loss_function.m, contains the source code. The python code will be uploaded soon.
To use it, simply place the dfl_loss_function.m in your current working MATLAB directory, or alternatively, add the location of dfl_loss_function.m in your MATLAB path.
Then, create the "classification layer" object in your neural network code as follows:

the_classification_layer = dfl_loss_function('<give some name>');
  
  
There are some loss control parameters associated with DFL, which you can tune from the backwardLoss() function inside dfl_loss_function.m. In the next update, these loss control parameters will be made as arguments of dfl_loss_function().
