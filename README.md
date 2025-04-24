# Code related to [link to publication]()

This code generates the figures related to the published paper [Enhancing Fin Whale Vocalizations in Distributed Acoustic Sensing Data](). It serves as a tutorial for the fin whale calls denoising pipeline described in the paper. 

## Python environment and [das4whales](https://das4whales.readthedocs.io/en/latest/src/install.html) installation
In command line, create a virtual environment for running the code:

```shell
python -m venv venv
```

Activate the environment 

```shell
source venv/bin/activate
```

Install `das4whales` and its dependencies

```shell
pip install 'git+https://github.com/DAS4Whales/DAS4Whales'
```

## Makefile 
All the scripts can be run using the `makefile`. Example for the section 3:

```shell
make section3
```