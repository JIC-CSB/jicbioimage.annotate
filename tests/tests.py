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
        import jicbioimage.annotate
        self.assertTrue(isinstance(jicbioimage.annotate.__version__, str))


class AnnotationCanvasUnitTests(unittest.TestCase):

    def test_blank_canvas(self):
        from jicbioimage.annotate import AnnotationCanvas
        canvas = AnnotationCanvas.blank_canvas(x=4, y=5)
        self.assertTrue(isinstance(canvas, AnnotationCanvas))
        self.assertTrue(isinstance(canvas, np.ndarray))
        self.assertEqual(canvas.shape, (4, 5, 3))
        self.assertEqual(canvas.dtype, np.uint8)
        self.assertEqual(np.sum(canvas), 0)

    def test_draw_cross(self):
        from jicbioimage.annotate import AnnotationCanvas
        canvas = AnnotationCanvas.blank_canvas(x=3, y=3)
        canvas.draw_cross(x=1, y=1, color=(1, 1, 1), radius=1)
        layer = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        expected = np.dstack([layer, layer, layer])
        self.assertTrue(np.array_equal(canvas, expected))

    def test_draw_cross_in_upper_left_corner(self):
        from jicbioimage.annotate import AnnotationCanvas
        canvas = AnnotationCanvas.blank_canvas(x=3, y=3)
        canvas.draw_cross(x=0, y=0, color=(1, 1, 1), radius=1)
        layer = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.uint8)
        expected = np.dstack([layer, layer, layer])
        self.assertTrue(np.array_equal(canvas, expected))

    def test_draw_cross_in_lower_right_corner(self):
        from jicbioimage.annotate import AnnotationCanvas
        canvas = AnnotationCanvas.blank_canvas(x=3, y=3)
        canvas.draw_cross(x=2, y=2, color=(1, 1, 1), radius=1)
        layer = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1]], dtype=np.uint8)
        expected = np.dstack([layer, layer, layer])
        self.assertTrue(np.array_equal(canvas, expected))

    def test_cross_outside_canvas_raises_index_error(self):
        from jicbioimage.annotate import AnnotationCanvas
        canvas = AnnotationCanvas.blank_canvas(x=3, y=3)
        with self.assertRaises(IndexError):
            canvas.draw_cross(x=3, y=1, color=(1, 1, 1), radius=1)
        with self.assertRaises(IndexError):
            canvas.draw_cross(x=1, y=3, color=(1, 1, 1), radius=1)

    def test_from_grayscale(self):
        from jicbioimage.annotate import AnnotationCanvas as AC
        grayscale = np.array([
            [0, 10, 20],
            [30, 40, 50],
            [60, 70, 80]], dtype=np.uint8)
        zeros = np.zeros((3, 3), dtype=np.uint8)

        gray_expected = np.dstack([grayscale, grayscale, grayscale])
        red_expected = np.dstack([grayscale, zeros, zeros])
        cyan_expected = np.dstack([zeros, grayscale, grayscale])

        gray_canvas = AC.from_grayscale(grayscale)
        self.assertTrue(np.array_equal(gray_canvas, gray_expected))

        red_canvas = AC.from_grayscale(grayscale, (True, False, False))
        self.assertTrue(np.array_equal(red_canvas, red_expected))

        cyan_canvas = AC.from_grayscale(grayscale, (False, True, True))
        self.assertTrue(np.array_equal(cyan_canvas, cyan_expected))


class FunctionalTests(unittest.TestCase):

    def setUp(self):
        if not os.path.isdir(TMP_DIR):
            os.mkdir(TMP_DIR)

    def tearDown(self):
        shutil.rmtree(TMP_DIR)

    def test_create_annotation_image_from_scratch(self):
        from jicbioimage.annotate import AnnotationCanvas

        # Create an empty canvas.
        canvas = AnnotationCanvas.blank_canvas(50, 75)

        # Draw a cross on it, centered on pixel 10,15.
        canvas.draw_cross(x=10, y=15, radius=1, color=(1, 0, 0))

        self.assertEqual(np.sum(canvas), 5)
        self.assertEqual(canvas[10, 15, 0], 1)
        self.assertEqual(canvas[9, 15, 0], 1)
        self.assertEqual(canvas[11, 15, 0], 1)
        self.assertEqual(canvas[10, 14, 0], 1)
        self.assertEqual(canvas[10, 16, 0], 1)


if __name__ == "__main__":
    unittest.main()
