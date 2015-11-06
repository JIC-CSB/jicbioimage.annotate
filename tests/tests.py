import unittest
import os
import os.path
import shutil

import numpy as np

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, 'data')
TMP_DIR = os.path.join(HERE, 'tmp')


class UnitTests(unittest.TestCase):

    def test_package_has_version_string(self):
        import jicbioimage.illustrate
        self.assertTrue(isinstance(jicbioimage.illustrate.__version__, str))

    def test_pretty_color(self):

        from jicbioimage.illustrate import pretty_color

        color = pretty_color()

        self.assertEqual(len(color), 3)
        self.assertTrue(isinstance(color, tuple))

        for _ in range(1000):
            color = pretty_color()
            self.assertTrue(all(0 <= c <= 255 for c in color))


class CanvasUnitTests(unittest.TestCase):

    def test_blank_canvas(self):
        from jicbioimage.illustrate import Canvas
        canvas = Canvas.blank_canvas(width=4, height=5)
        self.assertTrue(isinstance(canvas, Canvas))
        self.assertTrue(isinstance(canvas, np.ndarray))
        self.assertEqual(canvas.shape, (5, 4, 3))
        self.assertEqual(canvas.dtype, np.uint8)
        self.assertEqual(np.sum(canvas), 0)

    def test_draw_cross(self):
        from jicbioimage.illustrate import Canvas
        canvas = Canvas.blank_canvas(width=3, height=3)
        canvas.draw_cross(x=1, y=1, color=(1, 1, 1), radius=1)
        layer = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        expected = np.dstack([layer, layer, layer])
        self.assertTrue(np.array_equal(canvas, expected))

    def test_draw_cross_in_upper_left_corner(self):
        from jicbioimage.illustrate import Canvas
        canvas = Canvas.blank_canvas(width=3, height=3)
        canvas.draw_cross(x=0, y=0, color=(1, 1, 1), radius=1)
        layer = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.uint8)
        expected = np.dstack([layer, layer, layer])
        self.assertTrue(np.array_equal(canvas, expected))

    def test_draw_cross_in_lower_right_corner(self):
        from jicbioimage.illustrate import Canvas
        canvas = Canvas.blank_canvas(width=3, height=3)
        canvas.draw_cross(x=2, y=2, color=(1, 1, 1), radius=1)
        layer = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1]], dtype=np.uint8)
        expected = np.dstack([layer, layer, layer])
        self.assertTrue(np.array_equal(canvas, expected))

    def test_cross_outside_canvas_raises_index_error(self):
        from jicbioimage.illustrate import Canvas
        canvas = Canvas.blank_canvas(width=3, height=3)
        with self.assertRaises(IndexError):
            canvas.draw_cross(x=3, y=1, color=(1, 1, 1), radius=1)
        with self.assertRaises(IndexError):
            canvas.draw_cross(x=1, y=3, color=(1, 1, 1), radius=1)

    def test_mask_region(self):
        from jicbioimage.illustrate import Canvas
        canvas = Canvas.blank_canvas(3, 3)
        region = np.zeros((3, 3), dtype=bool)
        region[1, 1] = True
        canvas.mask_region(region, color=(0, 1, 0))
        self.assertEqual(np.sum(canvas), 1)
        self.assertTrue(canvas[1, 1, 1])

    def test_text_at_antialias(self):
        from jicbioimage.illustrate import Canvas
        canvas = Canvas.blank_canvas(width=6, height=6)
        layer = np.array(
            [[0,  38, 209, 233,  75, 0],
             [0, 182,  77,  48, 212, 0],
             [0, 239, 255, 255, 249, 0],
             [0, 239,  21,   0,   0, 0],
             [0, 176, 145,  10,   0, 0],
             [0,  31, 192, 246, 213, 0]], dtype=np.uint8)
        expected = np.dstack([layer, layer, layer])
        canvas.text_at("e", 0, 0, color=(255, 255, 255))
        self.assertTrue(np.array_equal(canvas, expected))

    def test_text_at_antialias_false(self):
        from jicbioimage.illustrate import Canvas
        canvas = Canvas.blank_canvas(width=6, height=6)
        layer = np.array([
            [0, 0, 1, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 0]], dtype=np.uint8)
        expected = np.dstack([layer, layer, layer])
        canvas.text_at("e", 0, 0, color=(1, 1, 1), antialias=False)
        self.assertTrue(np.array_equal(canvas, expected))

    def test_text_at_outside_image(self):
        from jicbioimage.illustrate import Canvas
        canvas = Canvas.blank_canvas(width=6, height=6)
        layer = np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0]], dtype=np.uint8)
        expected = np.dstack([layer, layer, layer])
        canvas.text_at("e", 0, 1, color=(1, 1, 1), antialias=False)
        self.assertTrue(np.array_equal(canvas, expected))

    def test_text_at_center_option(self):
        from jicbioimage.illustrate import Canvas
        canvas = Canvas.blank_canvas(width=6, height=6)
        layer = np.array([
            [0, 0, 1, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 0]], dtype=np.uint8)
        expected = np.dstack([layer, layer, layer])
        canvas.text_at("e", 3, 3, color=(1, 1, 1),
                       antialias=False, center=True)
        self.assertTrue(np.array_equal(canvas, expected))


class AnnotatedImage(unittest.TestCase):

    def test_from_grayscale(self):
        from jicbioimage.illustrate import AnnotatedImage as AnnIm
        grayscale = np.array([
            [0, 10, 20],
            [30, 40, 50],
            [60, 70, 80]], dtype=np.uint8)
        zeros = np.zeros((3, 3), dtype=np.uint8)

        gray_expected = np.dstack([grayscale, grayscale, grayscale])
        red_expected = np.dstack([grayscale, zeros, zeros])
        cyan_expected = np.dstack([zeros, grayscale, grayscale])

        gray_canvas = AnnIm.from_grayscale(grayscale)
        self.assertTrue(np.array_equal(gray_canvas, gray_expected))

        red_canvas = AnnIm.from_grayscale(grayscale, (True, False, False))
        self.assertTrue(np.array_equal(red_canvas, red_expected))

        cyan_canvas = AnnIm.from_grayscale(grayscale, (False, True, True))
        self.assertTrue(np.array_equal(cyan_canvas, cyan_expected))


class FunctionalTests(unittest.TestCase):

    def setUp(self):
        if not os.path.isdir(TMP_DIR):
            os.mkdir(TMP_DIR)

    def tearDown(self):
        shutil.rmtree(TMP_DIR)

    def test_create_annotation_image_from_scratch(self):
        from jicbioimage.illustrate import Canvas

        # Create an empty canvas.
        canvas = Canvas.blank_canvas(50, 75)

        # Draw a cross on it, centered on pixel 10,15.
        canvas.draw_cross(x=15, y=10, radius=1, color=(1, 0, 0))

        self.assertEqual(np.sum(canvas), 5)
        self.assertEqual(canvas[10, 15, 0], 1)
        self.assertEqual(canvas[9, 15, 0], 1)
        self.assertEqual(canvas[11, 15, 0], 1)
        self.assertEqual(canvas[10, 14, 0], 1)
        self.assertEqual(canvas[10, 16, 0], 1)


if __name__ == "__main__":
    unittest.main()
