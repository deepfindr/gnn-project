# Comments about the code
This is the code for this video series: https://www.youtube.com/watch?v=nAEb1lOf_4o

## Installing RDKIT
You will need rdkit to run this code.

Follow these instructions to install rdkit.
https://www.rdkit.org/docs/Install.html

If you run on Ubuntu / WSL you can simply run:
```
sudo apt-get install python-rdkit
```
Ideally execute the code in an anaconda environment, that's the easiest solution with rdkit.

## Installing the other packages
For pytorch geometric follow this tutorial:
https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

Make sure your CUDA version as well as torch version match the PyG version you install.
I've used torch 1.6.0 as it seemed to be most stable with the other libraries.

## Further things
- Its highly recommended to setup a GPU (including CUDA) for this code. 
- Here is where I found ideas for node / edge features: https://www.researchgate.net/figure/Descriptions-of-node-and-edge-features_tbl1_339424976
- There is also a Kaggle competition that used this dataset (from a University):
https://www.kaggle.com/c/iml2019/overview