# autoseg

[![](https://img.shields.io/pypi/pyversions/mwatershed.svg)](https://pypi.python.org/pypi/mwatershed)
[![](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)


## A package for automated deep-learning 3D image segmentation algorithms for connectomic brain mapping. Developed for X-Ray Nanoholography data.




* Free software: Apache 2.0 License

### Installation

Install Rust and Cargo via RustUp:

```bash
curl https://sh.rustup.rs -sSf | sh
```


Install MongoDB:

```bash
curl -fsSL https://pgp.mongodb.com/server-6.0.asc | sudo gpg -o /usr/share/keyrings/mongodb-server-6.0.gpg --dearmor
```


And initialize a MongoDB server in a screen on your machine:

```bash
screen
mongod
```

Install ``graph_tool``

```bash
conda install -c conda-forge -c ostrokach-forge -c pkgw-forge graph-tool
```


Install `autoseg`:

```bash
pip install git+https://github.com/brianreicher/autoseg.git
```

### Features


### Usage


### Credits

This package builds upon segmentation techniques & algorithms developed by Arlo Sheridan, Brian Reicher, Jeff Rhoades, Vijay Venu, and William Patton.
