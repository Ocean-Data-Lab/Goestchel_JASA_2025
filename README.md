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

## Scripts description 
The scripts in this repository are related to the sections of the paper [Enhancing Fin Whale Vocalizations in Distributed Acoustic Sensing Data]() and follow its organization. They depend on functions developed in [DAS4whales](https://github.com/DAS4Whales/DAS4Whales) and show different denoising techniques for 20 Hz fin whale vocalizations, on a 60s subset of data. Namely:
- `main_section3.py` shows the use of bandpass filtering, hybrid f-k filtering and SNR estimation.
- `main_section4.py` shows the effect of match-filtering and Gabor filtering on the data.
- `main_section4c.py` shows the effect of the noise envelope subtraction technique.
- `main_section5.py` shows the results of the time picking method for f-k filtered data and denoised data.

## DATA 

The data used in this code comes from the 2021 OOI RCA dataset:

>Wilcock, W., & Ocean Observatories Initiative. (2023). Rapid: A Community Test of Distributed Acoustic Sensing on the Ocean Observatories Initiative Regional Cabled Array [Data set]. Ocean Observatories Initiative. https://doi.org/10.58046/5J60-FJ89

The codes are set to fetch the data automatically from the OOI server. 

## Warning: high RAM usage
The scripts are memory intensive, and at least 32GB of RAM is recommended. Otherwise, the number of channels:

-`selected_channels_m_north = [12000, 66000, 5]`

-`selected_channels_m_north = [12000, 66000, 5]`

can be reduced by increasing the channel spacing in meters. The channel spacing of the raw data is 2m, the channel spacing used in these scripts is 4m and can be changed by increasing the last value in the lists above (e.g. 5). 