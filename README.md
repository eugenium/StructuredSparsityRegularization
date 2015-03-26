This is an implementation of a structured sparsity regularization method with total variation and l2 constraints . This is a strictly tighter approximation of sparsity +l2 + total variation than commonly used  elastic net penalty + Total variation. 

The example implementation includes a regularization of hinge loss. However a similar approach can be used with any loss function as long as the gradients can be provided.  

A version with much better optimization and support for the more commonly used isotropic TV will be released shortly. If interested contact me for preliminary access at eugene.belilovsky@inria.fr. 

Run test.m or test.py to get started
