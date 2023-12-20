import numpy as np
from sklearn.cluster import MiniBatchKMeans


COLOR_DICT = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "gray": (128, 128, 128),
    "silver": (192, 192, 192),
    "maroon": (128, 0, 0),
    "red": (255, 0, 0),
    "purple": (128, 0, 128),
    "fuchsia": (255, 0, 255),
    "green": (0, 128, 0),
    "lime": (0, 255, 0),
    "olive": (128, 128, 0),
    "yellow": (255, 255, 0),
    "navy": (0, 0, 128),
    "blue": (0, 0, 255),
    "teal": (0, 128, 128),
    "aqua": (0, 255, 255),
    "orange": (255, 165, 0),
    "pink": (255, 192, 203),
    "brown": (165, 42, 42),
    "tan": (210, 180, 140),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "violet": (238, 130, 238),
    "gold": (255, 215, 0),
    "turquoise": (64, 224, 208),
    "darkgreen": (0, 100, 0),
    "darkblue": (0, 0, 139),
    "firebrick": (178, 34, 34),
    "darkslategray": (47, 79, 79),
    "indigo": (75, 0, 130),
    "mediumvioletred": (199, 21, 133),
    "lightcoral": (240, 128, 128),
    "darkorange": (255, 140, 0),
    "yellowgreen": (154, 205, 50),
    "cadetblue": (95, 158, 160),
    "cornflowerblue": (100, 149, 237),
    "darkorchid": (153, 50, 204),
    "darksalmon": (233, 150, 122),
    "lightpink": (255, 182, 193),
    "lightskyblue": (135, 206, 250),
    "mediumpurple": (147, 112, 219),
    "seagreen": (46, 139, 87),
    "sienna": (160, 82, 45),
    "skyblue": (135, 206, 235),
    "slateblue": (106, 90, 205),
    "springgreen": (0, 255, 127),
    "darkcyan": (0, 139, 139),
    "chocolate": (210, 105, 30),
    "hotpink": (255, 105, 180),
    "limegreen": (50, 205, 50),
    "rosybrown": (188, 143, 143),
    "mediumseagreen": (60, 179, 113),
    "darkgoldenrod": (184, 134, 11),
    "darkviolet": (148, 0, 211),
    "saddlebrown": (139, 69, 19),
    "darkturquoise": (0, 206, 209),
    "orchid": (218, 112, 214),
    "palevioletred": (219, 112, 147),
    "mediumturquoise": (72, 209, 204),
    "mediumslateblue": (123, 104, 238),
    "palegreen": (152, 251, 152),
    "royalblue": (65, 105, 225),
    "darkolivegreen": (85, 107, 47),
    "indianred": (205, 92, 92),
    "darkmagenta": (139, 0, 139),
    "peru": (205, 133, 63),
    "darkseagreen": (143, 188, 143),
    "mediumaquamarine": (102, 205, 170),
    "mediumspringgreen": (0, 250, 154),
    "lightseagreen": (32, 178, 170),
    "steelblue": (70, 130, 180),
    "lightgreen": (144, 238, 144),
    "mediumorchid": (186, 85, 211),
    "plum": (221, 160, 221),
    "lightblue": (173, 216, 230),
    "lawngreen": (124, 252, 0),
    "burlywood": (222, 184, 135)
}


class Color:
    """
    Class representing a color utility.

    **Args:**
    - `color_dict (Dict, optional)`: Dictionary that maps common color names to their RGB tuple. Defaults to COLOR_DICT.
    - `num_clusters (int, optional)`: Number of clusters to use for color quantization. Defaults to 1.
"""
    def __init__(self, color_dict=None, num_clusters=1):
        if color_dict is None:
            color_dict = COLOR_DICT
        self.color_dict = color_dict
        self.num_clusters = num_clusters
        
    def find_closest_color(self, rgb_color, color_dict=None):
        """
        Finds the closest color name in the dictionary to the given RGB color.

        **Args:**
        - `rgb_color (Tuple)`: Color in RGB format.
        - `color_dict (Dict, optional)`: Dictionary that maps common color names to their RGB tuple. Defaults to None.

        **Returns:**
        - `String`: Closest color name.
"""
        
        if color_dict is None:
            color_dict = self.color_dict

        min_distance = float('inf')
        closest_color = None

        for name, color in color_dict.items():
            # Calculate Euclidean distance
            distance = np.linalg.norm(np.array(rgb_color) - np.array(color))

            # Check if this color is closer than the previous closest color
            if distance < min_distance:
                min_distance = distance
                closest_color = name

        return closest_color
    
    def get_dominant_color(self, roi, num_clusters=None):
        """
        Computes the dominant color among the pixels in the given region of interest (ROI).

        **Args:**
        - `roi (numpy.ndarray)`: Input region of interest.
        - `num_clusters (int, optional)`: Number of clusters to use for color quantization. Defaults to None.

        **Returns:**
        - `numpy.ndarray`: Dominant color(s) in RGB format."""
        if num_clusters is None:
            num_clusters = self.num_clusters

        # Reshape ROI pixels for color quantization
        pixels = roi.reshape(-1, 3) * 255
        
        # Apply MiniBatchKMeans clustering for color quantization
        kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=0, batch_size=256*4, init="k-means++", n_init=2)
        kmeans.fit(pixels)

        # Get the cluster centers (dominant colors)
        return kmeans.cluster_centers_.astype(int)

        

