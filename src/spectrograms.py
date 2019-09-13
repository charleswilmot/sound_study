from os.path import splitext
from pydub import AudioSegment
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import as_strided


mpl_events = ['button_press_event', 'button_release_event', 'draw_event', 'key_press_event',
              'key_release_event', 'motion_notify_event', 'pick_event', 'resize_event', 'scroll_event',
              'figure_enter_event', 'figure_leave_event', 'axes_enter_event', 'axes_leave_event']


class Spectrogram:
    def __init__(self, ax, path, resolution=(320, 240), window_size=2205):
        self._audiosegment = AudioSegment.from_file(path, format=splitext(path)[1])
        if self._audiosegment.channels == 2:
            self._audiosegment, _ = self._audiosegment.split_to_mono()
        self._frame_rate = self._audiosegment.frame_rate
        # self._data = self._audiosegment.raw_data
        self._data = np.float64(self._audiosegment.get_array_of_samples()) / 2**15
        self._ax = ax
        self._fig = self._ax.figure
        self._resolution = resolution
        self._window_size = window_size
        self._clip_value = 50
        self._hann_window = np.hanning(self._window_size)[:, np.newaxis] * 0.5 + 0.5
        # todo: read from data
        t_max = len(self.data) * 1000 / self.frame_rate
        f_min = self.frame_rate / self._window_size
        f_max = self.frame_rate / 2
        self._image = self._ax.imshow(
            # np.random.uniform(0, 1, size=self._resolution),
            self.analyse(0, t_max, f_min, f_max),
            extent=(0, t_max, 0, f_max),
            aspect="auto")
        self._xlim = 0
        self._ylim = 0
        # color maps:
        self._cmap_index = 0
        self._cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
        self._cid_zoom = self._fig.canvas.mpl_connect('draw_event', self._zoom_event_handler)
        self._cid_click = self._fig.canvas.mpl_connect('button_release_event', self._click_event_handler)
        self._cid_scroll = self._fig.canvas.mpl_connect("scroll_event", self._scroll_event_handler)

    def _lims_changed(self):
        xlim = self._ax.get_xlim()
        ylim = self._ax.get_ylim()
        ret = xlim != self._xlim or ylim != self._ylim
        return ret

    def _update_lims(self):
        self._xlim = self._ax.get_xlim()
        self._ylim = self._ax.get_ylim()

    def _get_resolution(self):
        return self._resolution

    def _get_freq_resolution(self):
        return self._resolution[1]

    def _get_time_resolution(self):
        return self._resolution[0]

    resolution = property(_get_resolution)
    freq_resolution = property(_get_freq_resolution)
    time_resolution = property(_get_time_resolution)

    def _get_data(self):
        return self._data

    data = property(_get_data)

    def _get_frame_rate(self):
        return self._frame_rate

    frame_rate = property(_get_frame_rate)

    def _zoom_event_handler(self, event):
        if self._lims_changed():
            self._update_lims()
            self._fig.canvas.mpl_disconnect(self._cid_zoom)
            self._image.set_data(self.analyse(*self._xlim, *self._ylim))
            self._image.set_extent([*self._xlim, *self._ylim])
            self._fig.canvas.draw()
            self._cid_zoom = self._fig.canvas.mpl_connect('draw_event', self._zoom_event_handler)

    def _scroll_event_handler(self, event):
        if event.button == "up":
            if self._clip_value < 5:
                self._clip_value += 0.1
            else:
                self._clip_value += 5
        if event.button == "down":
            if self._clip_value <= 5:
                if self._clip_value > 0.2:
                    self._clip_value -= 0.1
            else:
                self._clip_value -= 5
        self._image.set_clim(0, self._clip_value)
        self._fig.canvas.draw()
        print("New clip value: {: .1f}".format(self._clip_value))

    def _get_cmap_rotate(self):
        self._cmap_index += 1
        self._cmap_index %= len(self._cmaps)
        return self._cmaps[self._cmap_index]

    _cmap_rotate = property(_get_cmap_rotate)

    def _click_event_handler(self, event):
        if event.button == 1 and not self._lims_changed():
            cmap = self._cmap_rotate
            print("New color map: ", cmap)
            self._image.set_cmap(cmap)
            self._fig.canvas.draw()

    def time_to_index(self, t):
        return int((t * self.frame_rate) // 1000)

    # def time_to_slice(self, a=None, b=None):
    #     return slice(self.time_to_index(a), self.time_to_index(b))

    def fourier_matrix(self, f_min, f_max):
        angle = -2 * np.pi / self._window_size
        angles = np.full(shape=(self.freq_resolution, self._window_size), fill_value=angle)
        angles *= np.arange(self._window_size)[np.newaxis]
        maxi = f_max * self._window_size / self.frame_rate
        mini = f_min * self._window_size / self.frame_rate
        angles = angles * np.linspace(maxi, mini, self.freq_resolution)[:, np.newaxis]
        return np.exp(1j * angles) / np.sqrt(self._window_size)

    def strided_data(self, t_min, t_max, f_min, f_max):
        # shape = (self._window_size, self.time_resolution)
        # n_samples = self.frame_rate * (t_max - t_min) / 1000
        # strides = (self.data.itemsize * int((n_samples - self._window_size) // (self.time_resolution - 1)), self.data.itemsize)
        ret = np.zeros((self._window_size, self.time_resolution))
        start_sample = t_min * self.frame_rate / 1000
        end_sample = t_max * self.frame_rate / 1000 - self._window_size
        for i, float_sample in enumerate(np.linspace(start_sample, end_sample, self.time_resolution)):
            sample_index = int(float_sample)
            ret[:, i] = self.data[sample_index:sample_index + self._window_size]
        return ret

        # end_sample = start_sample + int(strides[0] * self.time_resolution / self.data.itemsize)
        # return np.copy(as_strided(self.data[start_sample:end_sample], shape, strides, writeable=False))

    def analyse(self, t_min, t_max, f_min, f_max):
        print("New range: time={: .2f}:{: .2f}\tfreq={: .2f}:{: .2f}".format(t_min, t_max, f_min, f_max))
        W = self.fourier_matrix(f_min, f_max)
        x = self.strided_data(t_min, t_max, f_min, f_max)
        return np.clip(np.abs(np.matmul(W, x * self._hann_window)), 0, self._clip_value)
        # return np.clip(np.abs(np.matmul(W, x)), 0, self._clip_value)


class DoubleFourierSpectrogram(Spectrogram):
    def __init__(self, ax, path, resolution=(320, 240), window_size=2205, second_window=[150, 2000]):
        self._second_window = second_window
        super().__init__(ax, path, resolution=resolution, window_size=window_size)

    def second_fourier_matrix(self, f_min, f_max):
        angle = -2 * np.pi / self.freq_resolution
        angles = np.full(shape=(self.freq_resolution, self.freq_resolution), fill_value=angle)
        angles *= np.arange(self.freq_resolution)[np.newaxis]
        maxi = f_max * self.freq_resolution / self.frame_rate
        mini = f_min * self.freq_resolution / self.frame_rate
        # maxi = f_max * self.freq_resolution
        # mini = f_min * self.freq_resolution
        angles = angles * np.linspace(maxi, mini, self.freq_resolution)[:, np.newaxis]
        return np.exp(1j * angles) / np.sqrt(self.freq_resolution)

    def analyse(self, t_min, t_max, f_min, f_max):
        print("New range: time={: .2f}:{: .2f}\tfreq={: .2f}:{: .2f}".format(t_min, t_max, f_min, f_max))
        W = self.fourier_matrix(*self._second_window)
        x = self.strided_data(t_min, t_max, *self._second_window)
        first = np.abs(np.matmul(W, x * self._hann_window)) # self.freq_resolution, self.time_resolution
        # W = freq_resolution, freq_resolution
        W = self.second_fourier_matrix(f_min, f_max)
        second = np.abs(np.matmul(W, first))
        return np.clip(second, 0, self._clip_value)


if __name__ == "__main__":
    import argparse


    parser = argparse.ArgumentParser()

    parser.add_argument(
        'path', metavar="PATH",
        type=str,
        action='store',
        help="Path to the wav."
    )

    path = parser.parse_args().path

    # plt.ion()
    # spect = Spectrogram(some_data)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    window_size = int(50 * 48000 / 1000)
    # spect = Spectrogram(ax, "../data/female_singing.wav", window_size=window_size, resolution=(2080, 720))
    # spect = Spectrogram(ax, path, window_size=window_size, resolution=(1080, 720))
    spect = DoubleFourierSpectrogram(ax, path, window_size=window_size, resolution=(1080, 720), second_window=[50, 1000])
    plt.show()
