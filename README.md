# ADRD
IJCAI2019 code for ADRD.

1.411Supplementary_experiment.pdf additional visual comparisons.
2.Code
(1) data_utils.py   dataprocessing file.
(2) Demo.py	    Code for test super-resolve images.
(3) model.pkl	    well-trained model of ADRD.
(4) Model2.py	    Model Graph file.
(5) Testdata	    Testing low-resolution images
(6) outdir	    output dir of super-resolved images and bicubic images

Environments: pytorch 0.4.0; Pillow 5.0.0; scikit-image 0.13.1; torchvision 0.2.0; python 2.7; numpy 1.14.0; scipy 1.0.0; At least double GPUs.

You can simply run “python Demo.py” for testing.

Download model at https://drive.google.com/open?id=1DpNkeT8KKDBq2dIHhbIuj01Sl2eQA3_F
