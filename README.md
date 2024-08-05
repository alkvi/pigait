# pigait - **P**ython **I**MU **gait** analysis library

![pylint](https://img.shields.io/badge/Pylint-9.80/10-green)
![flake8](https://img.shields.io/badge/flake8-checked-blueviolet)
![flake8](https://img.shields.io/badge/-Sphinx-000000?style=flat&logo=sphinx&logoColor=white)

This python library contains algorithms for spatiotemporal gait parameter calculation from 3-axis IMU/MARG data (accelerometer, magnetometer, gyroscope).

**If you are looking for a more complete project**, check out gaitmap: https://github.com/mad-lab-fau/gaitmap

This library is not hosted on PyPi / ReadTheDocs. If you want to use it, install it locally:

~~~
pip install -r pigait/requirements.txt
pip install -e pigait
~~~

To build the sphinx documentation:

~~~
cd pigait/docs
sphinx-apidoc.exe -o ./source ../src
~~~

Gait event detection methods:
- Wavelet analysis (based on [Pham et al 2017](https://doi.org/10.3389%2Ffneur.2017.00457))
- Peak detection on gyroscope data (based on [Salarian et al 2004](https://doi.org/10.1109/tbme.2004.827933) and [Mariani et al 2010](https://doi.org/10.1016/j.jbiomech.2010.07.003))

Spatial parameter calculation methods:
- Inverted pendulum model using lumbar data (based on [Del Din et al 2016](https://doi.org/10.1109/jbhi.2015.2419317))
- Strapdown inertial navigation / quaternion-based 3D position estimation with zero-velocity updates (ZUPT)

Three combinations have been validated against data simultaneously captured from an optoelectronic motion tracking system.
- Method 1 - Wavelet + inverted pendulum
- Method 2 - Peak detection + inverted pendulum
- Method 3 - Peak detection + strapdown inertial navigation

Results can be found in https://doi.org/10.1016/j.jbiomech.2023.111907 

TODO:
- Add examples on open data
- Add tests
- More algorithms etc

