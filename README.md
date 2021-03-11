# Hierarchical Supervised latent Dirichlet allocation for classification

(C) Copyright 2021, Written by Guanglei Yu, 114456486@qq.com, part of code is from Supervised latent Dirichlet allocation for classification.

This is a C++ implementation of Hierarchical sLDA for classification. 

## Sample data

A preprocessed 4-class hospital discharge records(HDRs) dataset from the First Affiliated Hospital of Xinjiang Medical University, China.



## References

[1] Chong Wang, David M. Blei and Li Fei-Fei. Simultaneous image classification and annotation. In CVPR, 2009. 

## README

Note that this code requires the Gnu Scientific Library, http://www.gnu.org/software/gsl/

------------------------------------------------------------------------


TABLE OF CONTENTS


A. COMPILING

B. ESTIMATION

C. INFERENCE


------------------------------------------------------------------------

A. COMPILING

Type "make" in a shell. Make sure the GSL is installed.

# To install GSL as follows:
# 1) download GSL: wget http://mirrors.ustc.edu.cn/gnu/gsl/gsl-1.9.tar.gz
# 2) tar zxvf gsl-1.9.tar.gz
# 3) cd gsl-1.9
# 4) ./configure
# 5) make
# 6) make install



------------------------------------------------------------------------

B. ESTIMATION

Estimate the model by executing:

     slda [est] [data] [data-fature] [label] [settings] [alpha] [k] [seeded/random/model_path] [directory]

The saved models are in two files:

     <iteration>.model is the model saved in the binary format, which is easy and
     fast to use for inference.

     <iteration>.model.txt is the model saved in the text format, which is
     convenient for printing topics or analysis using python.    

The variational posterior Dirichlets are in:

     <iteration>.gamma


Data format

(1) [data] is a file where each line is of the form:

     [M] [term_1]:[count] [term_2]:[count] ...  [term_K]:[count]

where [M] is the number of unique feature-value pair in the document, and the
[count] associated with each feature-value pair is how many times that term appeared
in the document. 

(2) [data-feature] is a file where each line is of the form:
     [N] [term_1]:[count] [term_2]:[count] ...  [term_K]:[count]
where [N] is the number of unique feature in the document, and the [count] associated with each feature is how many times that the corresponding feature-value pairs appeared in the document, which is equal to the sum of same feature's feature-value pairs.      
      

(3) [label] is a file where each line is the corresponding label for [data].
The labels must be 0, 1, ..., C-1, if we have C classes.


------------------------------------------------------------------------

C. INFERENCE

To perform inference on a different set of data (in the same format as
for estimation), execute:

    slda [inf] [data] [data-fature] [label] [settings] [model] [directory]
    
where [model] is the binary file from the estimation.
     
The target labels and predictive labels are in:

     inf-labels.dat

The variational posterior Dirichlets are in:

     inf-gamma.dat

     
## Copyright
Software provided as is under Gnu License.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
