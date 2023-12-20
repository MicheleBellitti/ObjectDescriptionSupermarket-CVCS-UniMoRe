import cv2
import numpy as np


class HomographyTransform:

    def __init__(self):
        pass

    def calculate_homography_and_transform(self, bbox1, bbox2):
        """

        Args:
            bbox1 ([Tuple]): first bounding box coordinates in format (x_min, y_min, x_max, y_max)
            bbox2 ([Tuple]): second bounding box coordinates in format (x_min, y_min, x_max, y_max)

        Returns:
            [cv2::UMat]: Result of homography transformation applied
        """
        src_points = self._extract_corners(bbox1)
        dst_points = self._extract_corners(bbox2)

        matrix, _ = cv2.findHomography(src_points, dst_points)
        transformed = cv2.perspectiveTransform(
            src_points.reshape(1, -1, 2), matrix)

        return transformed[0]

    def _extract_corners(self, bbox):
        """
        Function for extracting corners from a bounding box

        Args:
            bbox ([Tuple]): bounding box coordinates in format (x_min, y_min, x_max, y_max)

        Returns:
            [numpy.NDArray]: Array containing the coordinates of the 4 corners extracted
        """
        x_min, y_min, x_max, y_max = bbox
        return np.float32([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])
