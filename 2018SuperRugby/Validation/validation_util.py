import numpy as np

def extract_rgb(color):
    '''
    Extracts red, green, and blue components from color string
    
    Parameters
    ----------
    color (str):
        Color string. Must be #xxxxxx where xx is a 2-digit hexidecimal number

    Returns
    -------
    r (int):
        Amount of red
    g (int):
        Amount of green
    b (int):
        Amount of blue
    '''
    return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)

def fill10(hex_num):
    '''
    Addes zero in front of one-dimesional hexidecimal number e.g. a -> 0a. Returns input if 2-digit number is passed.
    
    Parameters
    ----------
    hex_num (str):
        Hexidecimal number

    Returns
    -------
    hex_num (str):
        Input number with leading zero added if necessary
    '''
    if len(hex_num) == 1:
        return '0' + hex_num
    else:
        return hex_num

def rgb2color(coords):
    '''
    Converts rgb coordinates in a tuple of integers to rgb string

    Parameters
    ----------
    coords (tup):
        Length-3 tuple indicating the red, green, and blue components of the color

    Returns
    -------
    color_string (str):
        Color string that can be used for plotting
    '''
    hex_r = fill10(hex(coords[0])[2:])
    hex_g = fill10(hex(coords[1])[2:])
    hex_b = fill10(hex(coords[2])[2:])

    return '#' + hex_r + hex_g + hex_b

def color_interpolate(start, end, N):
    '''
    Interpolates a gradient between two colors

    Parameters
    ----------
    start (str):
        Color string to start with
    end (str):
        Color string to end with
    N (int):
        Number of points to interpolate

    Returns
    -------
    gradient (list):
        Gradients of colors
    '''
    #Get red, green, and blue components of start and end colors
    (start_r, start_g, start_b) = extract_rgb(start)
    (end_r, end_g, end_b) = extract_rgb(end)

    #Create linear spaces between start and end components
    gradient_r = np.round(np.linspace(start_r, end_r, N)).astype(int)
    gradient_g = np.round(np.linspace(start_g, end_g, N)).astype(int)
    gradient_b = np.round(np.linspace(start_b, end_b, N)).astype(int)

    #Recombine into color gradient
    gradient = []
    for i in range(N):
        gradient.append(rgb2color((gradient_r[i], gradient_g[i], gradient_b[i])))

    return gradient