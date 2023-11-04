# Discrete wavelet transform (DWT) via lifting in PyTorch

This repository provides implementation of discrete wavelet transform (DWT) vis lifting scheme in PyTorch.
Operations can run on both: CPU and GPU, filter coefficients can be made trainable parameters of model.
The following wavelets are implemented (15 in total):
1) CDF-9/7 (used in lossy compression in JPEG 2000)
2) CDF-5/3 (used in lossless compression in JPEG 2000)
3) Haar
4) Biorthogonal Spline Wavelets: 3/3, 3/5, 3/7, 3/9, 4/8
5) Reverse Biorthogonal Spline Wavelets: 3/3, 3/5, 3/7, 3/9, 4/8
6) Daubechies: 4
7) Coiflets: 12

All the schemes have been tested for perfect reconstruction, i.e, operations of analysis and synthesis are invertible: 
the error varies from approximately `1e-15` to `1e-14`.

This repository also has a TensorFlow port: https://github.com/uladzislau-varabei/tf-wavelet-lifting.



## Results 

Below are some of the results with all the implemented wavelets. See file `tests.py` for details.

Textures:
![dwt_textures](results/wavelets_textures.png)

Some real world image sample:
![dwt_image](results/wavelets_castle2.png)



## DWT lifting scheme

To get a better understanding of what lifting scheme for wavelets is, refer to the following image:

![DWT_lifting](figs/DWT_Lifting.png)



## DWT for images

Application of DWT to 2d input (images) can be summarized with the following figure:

![DWT_2d](figs/DWT_2d.png)

The same approach is applied to each image channel and to each image in batch. 
The implementation in this repository is vectorized, that is, at the same time all columns or rows of input are used, 
so it's very fast. 



## Additional tests 

For clearer understanding of all the implemented wavelets it can be useful to see results on additional images. 
Such results are presented in the `results` directory. 
Original images for testing can be found in the `images` directory.



## Acknowledgements

For details on lifting schemes for different wavelets a special thank is for 
[Alex Malevich](https://scholar.google.com/citations?user=lQt-qqUAAAAJ&hl=eng).
