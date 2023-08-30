
""" a list of coords cartesian [(x, y), (x, y)] represent a polygon"""

import math
coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
# test point

for x, y in coords:
    # min_dist_point = math.inf
    # min_dist = None
    # for x1, y1 in coords[1:]:
    #     dist = math.sqrt((x - x1) ** 2 + (y - y1) ** 2)
    #     if dist < min_dist:
    #         min_dist = dist
    #         min_dist_point = (x1, y1)
    
    # check if any other point lies on their slope


