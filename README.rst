Maths and Music in Python
-------------------------

This is a Python package to experiment with audio files (wav files) and performs some math
operations over audio signals. The library can read files in different formats and convert
them into numpy array to work with them. The main functionalities consist in open audio
audio files, make plots in time and frequency domain, compute some features, and apply
 filter operations, finally all results can be saved into a wav file.

To use do the following:

>>> import mmpy as mm
>>> mm.style() # set a particular style
>>> a = mm.Audio(filename)
>>> a.plot_temp()
>>> a.plot_freq()

>>> b = mm.load(filename)
>>> b.plot_temp()
>>> b.plot_freq()

