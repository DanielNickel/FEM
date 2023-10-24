#----------ISD | TU Braunschweig----------#
#-----------Beethovenstraße 51------------#
#-----------38106 Braunschweig------------#
#-----------------------------------------#

import matplotlib as mpl
import numpy as np

def set_color_map(min, max):
    # Get color map 'viridis' and set color for out of bounds values
    color_map = mpl.colormaps['viridis']
    color_map.set_over('black')
    color_map.set_under('black')

    # if difference between min and max is too small add +-1
    if max - min < 0.01:
        min = np.around(min-1, 2)
        max = np.around(max+1, 2)
    color_limits = (min, max)

    return color_map, color_limits


def set_color_bar_ticks(ticks, limits):
    # Get the difference between min and max limit
    diff = limits[1] - limits[0]

    # Get the differences between min and max values of the color map and the color bar ticks
    min_diff = abs(limits[0] - ticks[0])
    max_diff = abs(limits[1] - ticks[-1])

    # if differences are considerable replace tick values for exact color map limits
    if min_diff > 1e-5 * diff:
        if min_diff < 0.5 * (ticks[1] - ticks[0]):
            ticks = np.insert(ticks[1:], 0, limits[0])
        else:
            ticks = np.insert(ticks[2:], 0, limits[0])
    if max_diff > 1e-5 * diff:
        if max_diff < 0.5 * (ticks[-1] - ticks[-2]):
            ticks = np.append(ticks[:-1], limits[1])
        else:
            ticks = np.append(ticks[:-2], limits[1])

    return ticks
