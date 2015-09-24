"""jicbioimage.annotate package."""

import numpy as np

__version__ = "0.0.1"


class AnnotationCanvas(np.ndarray):
    """Class for building up annotated images."""

    @staticmethod
    def blank_canvas(x, y):
        """Return a blank canvas to annotate.

        :param x: xdim
        :param y: ydim
        :returns: :class:`jicbioimage.annotate.AnnotationCanvas`
        """
        canvas = np.zeros((x, y, 3), dtype=np.uint8)
        return canvas.view(AnnotationCanvas)

    @staticmethod
    def from_grayscale(im, channels_on=(True, True, True)):
        """Return a canvas from a grayscale image.

        :param im: single channel image
        :channels_on: channels to populate with input image
        :returns: :class:`jicbioimage.annotate.AnnotationCanvas`
        """
        xdim, ydim = im.shape
        canvas = np.zeros((xdim, ydim, 3), dtype=np.uint8)
        for i, include in enumerate(channels_on):
            if include:
                canvas[:, :, i] = im
        return canvas.view(AnnotationCanvas)

    def draw_cross(self, x, y, color=(255, 0, 0), radius=4):
        """Draw a cross on the canvas.

        :param x: x coordinate (int)
        :param y: y coordinate (int)
        :param color: color of the cross (RGB tuple)
        :param radius: radius of the cross (int)
        """
        for xmod in np.arange(-radius, radius+1, 1):
            xpos = x + xmod
            if xpos < 0:
                continue  # Negative indices will draw on the opposite side.
            if xpos >= self.shape[0]:
                continue  # Out of bounds.
            self[xpos, y] = color
        for ymod in np.arange(-radius, radius+1, 1):
            ypos = y + ymod
            if ypos < 0:
                continue  # Negative indices will draw on the opposite side.
            if ypos >= self.shape[1]:
                continue  # Out of bounds.
            self[x, ypos] = color
