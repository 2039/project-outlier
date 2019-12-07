from matplotlib import pyplot as plt
from matplotlib import axes
from matplotlib.animation import FuncAnimation
from functools import partial

"""
This module is a wrapper for matplotlib to make plotting simpler

(matplotlib sucks btw)
"""



class _Ax:
    def __init__(self, fig):
        self._Fig = fig

    def __getitem__(self, coords):
        return self._ax(*coords)

    def _ax(self, x, y):
        self._Fig.axes[x, y] = self._Fig._fig.add_subplot(self._Fig._gs[x, y])
        return self._Fig.axes[x, y]


class Figure:
    def __init__(self, x, y):
        self._fig = plt.figure()
        self.gridspec = (x, y)
        self.ax = _Ax(self)
        self.axes = {}
        self._shown = False

    def __enter__(self):
        try:
            return self
        except Exception as e:
            print(e)

    def __exit__(self, type, value, traceback):
        try:
            if traceback is None:
                if not self._shown: self._fig.show(); self._fig.canvas.draw()
                self._shown = True
                input("Press any key to continue")
                plt.close()
            else:
                raise
        except:
            import sys
            if sys.exc_info()[1] is not value:
                raise


    def set_imshow(self, ax, array):
        ax.get_images()[0].set_data(array)

    @property
    def gridspec(self):
        return self._gs

    @gridspec.setter
    def gridspec(self, shape):
        self._gs = self._fig.add_gridspec(*shape)


    def show(self):
        if not self._shown: self._fig.show(); self._fig.canvas.draw()
        self._shown = True

    def pause(self, t):
        plt.pause(t)

    def update(self, t=0.001):
        self.pause(t)


    def animate(
            self,
            update_func,
            frames=100,
            repeat=True,
            repeat_delay=2000,
            init=lambda: None,
            interval=100, # frame length in ms
            blit=False,
        ):

        anim = FuncAnimation(
            fig=self._fig, func=update_func, frames=frames,
            repeat=repeat, repeat_delay=repeat_delay,
            init_func=init, interval=interval, blit=blit,
        )

        plt.show()
        plt.close()
