# MELD:  
Mixed Effects for Large Datasets

[![CircleCI](https://circleci.com/gh/compmem/MELD.svg?style=svg)](https://circleci.com/gh/compmem/MELD)

MELD uses rpy2 to wrap r's LME4 library. rpy2 is sometimes difficult to install, so we've provided a docker image with MELD, rpy2, and a jupyter notebook. Just run `docker run --rm -p 8888:8888 compmem/meld` to start MELDing.

If you use MELD, please cite our publication:
Nielson DM, Sederberg PB (2017) [MELD: Mixed effects for large datasets](https://doi.org/10.1371/journal.pone.0182797). PLOS ONE 12(8): e0182797. https://doi.org/10.1371/journal.pone.0182797

This repo is very much a work in progress, but MELD works in python 3, and is distributed with a docker container to alleviate the pain of rpy2 installation. 
