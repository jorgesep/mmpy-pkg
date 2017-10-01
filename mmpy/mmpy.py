import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

import scipy as sp
from scipy.io.wavfile import read
from scipy.fftpack import fft
# from scipy import signal

import soundfile as sf

import os

"""
Python package with the scientific library mmpy (Maths and Music in Python)
that works with audio files. This library can read audio files in different
formats, and plot its temporal and/or spectral audio representations, compute
some features, apply filters and save results on disk.
"""


# load audio file
def load(filename):
    """
    Function to load audio file from input path
    :return: Audio object
    """

    try:
        with open (filename) as f:
            print ("Opening audio file, please wait...")
            data, samplerate = sf.read (filename)

    except IOError as e:
        print (type (e).__name__ + ': ' + str (e))
        print ("Error: %s not found." % filename)
        raise
    except Exception as e:
        print (type (e).__name__ + ': ' + str (e))
        raise

    return Audio(data=data, samplerate=samplerate)

# function to change style
def style():
    # Switch seaborn to defaults ("darkgrid").
    sb.set ()

    # sb.set_palette("husl")

    # Settting to "whitegrid" theme, background color in ligth yellow
    sb.set_style ("whitegrid", {'grid.linestyle': '--', 'font.family': ['Arial'], 'axes.facecolor': '#f5fee2'})

    # Change font size, and plot lines width
    sb.set_context ("notebook", font_scale=1.5, rc={'lines.linewidth': 1.5})

    # sb.color_palette("BuGn_r")
    sb.set (rc={"figure.figsize": (10, 5)})

    current_palette_7 = sb.color_palette ("hls", 7)
    sb.set_palette (current_palette_7)

    print ('Setting plot local styles based on \'whitegrid\' ')


class Audio:
    """
    Class to operate over audio signals.

    The input is provided in the form of a audio file or a ndarray
    object. The input signal can be plotted either in time or frequency.
    """

    # Constructor
    def __init__(self, data=None, file=None, samplerate=None):
        """
        Constructor for this class.
        """

        # signal data
        self._data = None
        # set by default sample rate to 22.05 kHz
        self._rate = samplerate
        if samplerate is None: self._rate = 22050


        if data is not None:
            if not isinstance (data, np.ndarray):
                raise Exception ('Input array not numpy.ndarray type')
            self._data = data
        elif file is not None:
            self._rate, self._data = sp.io.wavfile.read (file, mmap=False)
        else:
            raise Exception ("No input data provided !.")

        # number of sample points
        self._nsamples = len (self._data)

        # gets duration of signal audio
        self._duration = (1. / self._rate) * self._nsamples

        # set time range
        self._init_time = 0
        self._end_time = self._duration

        # set freq range
        self._init_freq = 0
        self._end_freq = self._rate // 2

    @property
    def duration(self):
        return self._duration

    @property
    def sample_points(self):
        return self._nsamples

    @property
    def sample_rate(self):
        """
        Samples of audio carried per second.
        """
        return self._rate

    @sample_rate.setter
    def sample_rate(self, value):
        """
        Set sample rate: 8, 11.025, 22.05, 32, 44, 48, 64, 88.2, 96 kHz.
        """
        self._rate = value
        # change duration
        self._duration = (1. / self._rate) * self._nsamples

    @property
    def time_range(self):
        """
        takes init and end time to compute range of the samples.
        """
        a, b = [int (self._rate * i) for i in [self._init_time, self._end_time]]
        return range (a, b)

    @property
    def frequency_range(self):
        """
        takes init and end time to compute range of the samples.
        """
        # Sampling rate
        Fs = float (self._rate)
        # number of sample points
        N = len (self._data)

        # frequency resolution
        f = np.linspace (0, Fs / 2, N // 2)
        # df = Fs / N
        df = f[1]

        a, b = [int (i / df) for i in [self._init_freq, self._end_freq]]

        # removing DC component
        if a == 0: a = 1

        return range (a, b)

    def plot_time(self):
        """
        Time domain plotting of input audio signal.
        """

        # range taken from init_time & end_time local variables
        _range = self.time_range

        # Sampling rate
        rate = self._rate

        # number of sample points
        N = self._nsamples

        t = np.linspace (0, N / rate, num=N)

        f, ax = plt.subplots (ncols=1, nrows=1, num='plot_time')

        # By default plot the first 1024 samples
        ax.plot (t[_range], self._data[_range])

        # set title and labels on the graphs.
        ax.set_title ('Audio signal')
        # label the axes
        ax.set_xlabel ('Seconds')
        ax.set_ylabel ("Amplitude")

        # display the plot
        plt.show ()

    def set_time(self, trange=None):
        """
        Setting audio values for time representation.
        Input parameter: a.set_time([0.1,0.9])
        """
        if not ((isinstance (trange, np.ndarray) or isinstance (trange, list)) and len (trange) == 2):
            raise Exception ('no input argument!.')

        # get range of time in seconds
        self._init_time, self._end_time = trange

        # swap over values in case of error
        if self._init_time > self._end_time:
            self._init_time, self._end_time = trange[::-1]

    def plot_freq(self):
        """
        frequency domain plot using Fourier analysis
        """

        # Sampling rate
        Fs = self._rate

        # Sampling interval
        Ts = 1.0 / Fs

        # get time range from loca time variables
        f_range = self.frequency_range

        print (f_range)

        # audio time signal
        X = self._data

        # number of sample points
        N = self._nsamples

        # sample spacing
        T = 1.0 / self._rate

        # frequency resolution
        # df = np.float(Fs) / N

        # x-axis index converted to frequency
        # f = np.arange(N)*df

        # generates N/2 samples with a evenly spaced numbers over
        # [0, Fs/2] interval.
        f = np.linspace (0, Fs / 2, N // 2)

        # get PSD with fast fourier transform
        p = 2.0 / N * np.abs (sp.fftpack.fft (X))

        # get only positive frequency components
        # p = p[1:N // 2 - 1]
        # f = f[1:N // 2 - 1]
        p = p[f_range]
        f = f[f_range]

        print ('N:', N, 'N//2:', N // 2 - 1)

        plt.close ('plot_freq')
        # Create figure with only one subplot
        fig, ax = plt.subplots (ncols=1, nrows=1, num='plot_freq')

        # plot frequency domain, removing DC component, and Nyquist.
        # ax.plot(f[1:], p[1:N//2])
        ax.plot (f, p)

        # set title and labels on the graphs.
        # ax.set_title('Frequency')
        # label the axes
        ax.set_xlabel ('frequency [Hz]')
        ax.set_ylabel ("|amplitude|")

        # display frequency plot
        plt.show ()

    def set_freq(self, frange=None):
        """
        Setting audio values for time representation.
        Input parameter: a.set_time([0.1,0.9])
        """
        if not ((isinstance (frange, np.ndarray) or isinstance (frange, list)) and len (frange) == 2):
            raise Exception ('no input argument')

        # get range of time in seconds
        self._init_freq, self._end_freq = frange

        # swap over values in case of error
        if self._init_time > self._end_time:
            self._init_time, self._end_time = frange[::-1]

        if self._end_freq > (self._rate // 2):
            self._end_freq = self._rate // 2
        if self._init_time < 0 and self._init_time > self._end_freq:
            self._init_freq = 0

    def reset(self):
        """
        set up audio to its original values.
        """
        # set original time range
        self._init_time = 0
        self._end_time = self._duration

        # set original freq range
        self._init_freq = 0
        self._end_freq = self._rate // 2

        plt.clf ()
        plt.close ('all')

    def save(self, file=None):
        """
        Save audio representation to file as wav file.
        :param file: name of the file to save wav data
        :return: 
        """

        if file is None:
            print ('Insert name of the file to save audio wav file.')

            return

        # get period of time to be saved.
        _range = self.time_range

        # write numpy data array as a WAV file
        sp.io.wavfile.write (file, self._rate, self._data[_range])

    def play(self):
        """
        Play contents of the audio data signal representation. The play can
        be interrupted using Ctrl-C.

        Requires sounddevice module for this to work.
        (http://python-sounddevice.readthedocs.io/en/0.3.8/)

        :return:
        """

        try:
            # import module to play sound
            import sounddevice as sd

            # by default play only 5 seconds.

            # to get just a range of time to be played
            sound = self._data[self.time_range]
            samplerate = self.sample_rate

            sd.play (sound, samplerate=samplerate, blocking=True)
            status = sd.wait ()
            if status:
                print ('Error during playback: ' + str (status))

        except KeyboardInterrupt:
            print ('\nInterrupted by user')
        except Exception as e:
            print (type (e).__name__ + ': ' + str (e))

    def test_signal(self):
        """
        Method that generates time signal and plot
        in time and frquency domain.
        """

        Fs = 22050;  # sampling rate
        Ts = 1.0 / Fs;  # sampling interval
        t = np.arange (0, 1, Ts);  # time vector

        # Synthetic audio signal
        y = 1.5 * np.sin (2 * np.pi * 5000 * t) + 2.0 * np.sin (2 * np.pi * 7500 * t) + 3.4 * np.sin (
            2 * np.pi * 8000 * t) + 0.3 * np.sin (2 * np.pi * 50 * t) + np.random.randn (len (t))

        N = len (y);  # length of the signal
        k = np.arange (N)
        T = N / Fs
        frq = k / T;  # two sides frequency range
        frq = frq[range (N // 2)];  # one side frequency range

        Y = 2 / N * sp.fftpack.fft (y);  # fft computing and normalization
        Y = Y[range (N // 2)]

        # Create figure with only one subplot
        fig, (ax1, ax2) = plt.subplots (ncols=1, nrows=2, num='Signal Testing')
        ax1.plot (t, y)
        ax1.set_xlabel ('Time')
        ax1.set_ylabel ('Amplitude')
        ax2.plot (frq, abs (Y), 'r')  # plotting the spectrum
        ax2.set_xlabel ('Freq (Hz)')
        ax2.set_ylabel ('|Y(freq)|')

        # display frequency plot
        plt.show ()


if __name__ == '__main__':
    # audio file
    filename = 'swvader01.wav'
    # audio object
    a = Audio (file=filename)
    # plot time samples
    # a.plot_time()
    # a.set_time([0.5, 1.5])
    # a.plot_time()

    # a.plot_freq()
    # a.plot_freq1()
    # a.plot_freq2()

    a.test_signal ()
