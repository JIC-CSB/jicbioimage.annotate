"""Module for creating annotated images.

To create an annotated image we need an instance of the
:class:`jicbioimage.illustrate.Canvas` class.

>>> from jicbioimage.illustrate import Canvas

Suppose that we have an existing image.

>>> from jicbioimage.core.image import Image
>>> im = Image((50,50))

We can use this image to create an canvas instance populated with the data
as a RGB gray scale image.

>>> canvas = Canvas.from_grayscale(im)

The :class:`jicbioimage.illustrate.Canvas` instance has built in annotation
functionality. We can draw a cross at coordinates (10, 20).

>>> canvas.draw_cross(10, 20)

Or mask out a bitmap with the color cyan.

>>> bitmap = np.zeros((50, 50), dtype=bool)
>>> bitmap[30:40, 30:40] = True
>>> canvas.mask_region(bitmap, color=(0, 255, 255))

"""

import numpy as np

__version__ = "0.0.1"


class Canvas(np.ndarray):
    """Class for building up annotated images."""

    @staticmethod
    def blank_canvas(x, y):
        """Return a blank canvas to annotate.

        :param x: xdim
        :param y: ydim
        :returns: :class:`jicbioimage.illustrate.Canvas`
        """
        canvas = np.zeros((x, y, 3), dtype=np.uint8)
        return canvas.view(Canvas)

    @staticmethod
    def from_grayscale(im, channels_on=(True, True, True)):
        """Return a canvas from a grayscale image.

        :param im: single channel image
        :channels_on: channels to populate with input image
        :returns: :class:`jicbioimage.illustrate.Canvas`
        """
        xdim, ydim = im.shape
        canvas = np.zeros((xdim, ydim, 3), dtype=np.uint8)
        for i, include in enumerate(channels_on):
            if include:
                canvas[:, :, i] = im
        return canvas.view(Canvas)

    def draw_cross(self, x, y, color=(255, 0, 0), radius=4):
        """Draw a cross on the canvas.

        :param x: x coordinate (int)
        :param y: y coordinate (int)
        :param color: RGB tuple
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

    def mask_region(self, region, color=(0, 255, 0)):
        """Mask a region with a color.

        :param region: :class:`jicbioimage.core.region.Region`
        :param color: RGB tuple
        """
        self[region] = color
