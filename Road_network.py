import argparse
from zipfile import ZipFile
import numpy as np
from scipy.interpolate import interp1d
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import math
import sys
import time
from fastkml import kml
from fastkml.geometry import LineString
from fastkml import styles
from rdp import rdp

class KmlTrackReader():
    """Class to save all the Tracks in a kml file"""
    def __init__(self, filename):
        import pandas as pd
        import xmltodict
        kml_dict = xmltodict.parse(filename)
        doc = kml_dict['kml']['Document']
        self.tracks = []
        for Placemark in doc['Placemark']:
            tracks = Placemark['gx:MultiTrack']
            one_item = False
            for track in tracks['gx:Track']:
                if (type(track) is str): # only one gx:Track in gx:Multitrack (if not, it will return an error)
                    #self.tracks.append(self.Track(tracks['gx:Track']))
                    self.tracks.append([[np.float(coord.split(' ')[0]), np.float(coord.split(' ')[1])] for coord in tracks['gx:Track']['gx:coord']])
                    break
                #self.tracks.append(self.Track(track))
                self.tracks.append([[np.float(coord.split(' ')[0]), np.float(coord.split(' ')[1])] for coord in track['gx:coord']])

def KML_creator(tracks):
    result = kml.KML()
    ns = '{http://www.opengis.net/kml/2.2}'
    d = kml.Document(ns, '0', 'Road Network')
    f = kml.Folder(name='TRACKS')
    result.append(d)
    d.append(f)
    for i, track in enumerate(tracks):
        placemark = kml.Placemark(ns, str(i), str(i))
        style = styles.Style(styles=[styles.LineStyle(color='ff0000ff'), styles.PolyStyle(fill=0)])
        placemark.append_style(style)
        trajectory = []
        for point in track:
            trajectory.append(tuple(point))
        placemark.geometry = LineString(trajectory)
        f.append(placemark)
    return result

def Road_Network(dataset, filename, cell_size, resample_meters, equidistance_plots, plot_result, sigma, SlideEpsilon, ImgFilter, delete_edges, saveSlide):
    '''The main algorithm'''

    def meters_to_latitude(meters):
        '''A nautical mile is 1852 meters, that is one minute of arc at the meridian.
        So, in degrees there are 1852 * 60 = 111.120 aprox'''
        return meters / 111120

    def meters_to_longitude(meters, latitude):
        '''Depending on latitude. Along the equator there are aprox 40074784 meters.
        So, one degree of longitude at the equator is 111319 aprox'''
        return meters * abs(math.cos(latitude)) / 111319

    def linearEquidistanceTranformation(track):
        '''Transform a track into an linearly equidistant over the polygonal track'''
        x = [coord[0] for coord in track]
        y = [coord[1] for coord in track]

        # Linear length on the line
        distance = np.cumsum(np.sqrt( np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2 ))
        num_points = math.ceil(distance[-1] / meters_to_latitude(resample_meters))
        distance = distance/distance[-1]
        fx, fy = interp1d( distance, x ), interp1d( distance, y )

        alpha = np.linspace(0, 1, num_points)
        x_regular, y_regular = fx(alpha), fy(alpha)
        if equidistance_plots:
            plt.plot(x, y, 'o-')
            plt.plot(x_regular, y_regular, 'or')
            plt.axis('equal')
            plt.show()
        return list(list(coord) for coord in zip(x_regular, y_regular))

    minX = minY = sys.float_info.max
    maxX = maxY = sys.float_info.min
    tracks = []
    for track in dataset:
        new_track = linearEquidistanceTranformation(track)
        for coord in track:
            if (coord[0] < minX):
                minX = coord[0]
            elif (coord[0] > maxX):
                maxX = coord[0]
            if (coord[1] < minY):
                minY = coord[1]
            elif (coord[1] > maxY):
                maxY = coord[1]
        tracks.append([[np.float(point[0]),np.float(point[1])] for point in new_track])

    # It is an unprovable case but must be contemplated, especially if using split and merge with many partitions
    if not tracks:
        return [], []

    def grid_position_search(min_coord_X, cell_size_X, min_coord_Y, cell_size_Y, point):
        '''Finds in constant time the position of the point with respect 
        to the cell size and the minimum coordinate of the grid'''
        return math.floor((point[0] - min_coord_X) / cell_size_X), math.floor((point[1] - min_coord_Y) / cell_size_Y)

    def density_grid(cell_size, tracks, return_original, minX, minY, maxX, maxY):
        '''returns the initial density grid normalized'''
        def init_grid(cell_size):
            '''Initializes a grid with square cells of cell_size meters between [minX, minY] and [maxX, maxY]'''
            cell_size_lat = meters_to_latitude(cell_size)
            cell_size_lon = meters_to_longitude(cell_size, (minY + maxY) / 2)
            num_cells_X = math.ceil(abs(maxX - minX) / cell_size_lon)
            num_cells_Y = math.ceil(abs(maxY - minY) / cell_size_lat)
            return np.zeros((num_cells_X, num_cells_Y)), cell_size_lon, cell_size_lat

        grid, cell_size_lon, cell_size_lat = init_grid(cell_size)
        for track in tracks:
            for coord in track:
                x, y = grid_position_search(minX, cell_size_lon, minY, cell_size_lat, coord)
                grid[x, y] += 1
        # Apply a gaussian blur
        blurred = ndi.gaussian_filter(grid, sigma=sigma)
        if return_original:
            return blurred, cell_size_lon, cell_size_lat, grid
        else:
            return blurred, cell_size_lon, cell_size_lat

    Dgrid, cell_size_lon, cell_size_lat = density_grid(cell_size, tracks, False, minX, minY, maxX, maxY)

    # Uncomment this to visualize the density grid with a heat map
    fig, ax = plt.subplots()
    im = ax.imshow(Dgrid)

    ax.set_title("Density grid after Gaussian blur")
    fig.tight_layout()
    plt.show()

    def SlideMethod_adapted(grid, cell_sizeX, cell_sizeY, tracks):
        '''This is Slide method adapted to the algorithm in order to adjust
        trajectories to the denser parts'''
        def Surface_component(grid, cell_sizeX, cell_sizeY, point, minX, minY):
            '''This is the component to drag the point to the denser part. It is the
            gradient that is approximately estimated with a billinear interpolation'''
            def bilinear_interpolation(left_down, right_down, left_up, right_up, deltaX, deltaY):
                '''Given the four corners and their the values, calculates the bilinear interpolation of the point'''
                u1 = left_down * (1 - deltaX) + right_down * deltaX
                u2 = left_up * (1 - deltaX) + right_up * deltaX
                w1 = (1 - deltaY) * (right_down - left_down)
                w2 = deltaY * (right_up - left_up)
                result = np.array([w1 + w2, u2 - u1])
                
                lowerX = left_down * (1 - deltaX) + right_down * deltaX
                upperX = left_up * (1 - deltaX) + right_up * deltaX
                density = lowerX * (1 - deltaY) + upperX * deltaY
                return result, density

            # First, find the grid cells around the point
            x, y = grid_position_search(minX, cell_sizeX, minY, cell_sizeY, point)
            x = max(0, min(x, grid.shape[0]-1))
            y = max(0, min(y, grid.shape[1]-1))

            # We need to know exactly in which quadrant falls the point inside the cell
            cell_center_X = minX + (x + 0.5) * cell_sizeX
            cell_center_Y = minY + (y + 0.5) * cell_sizeY
            right = cell_center_X < point[0]
            up = cell_center_Y < point[1]
            deltaX = deltaY = indX0 = indX1 = indY0 = indY1 = 0
            if (right):
                deltaX = (point[0] - cell_center_X) / cell_sizeX
                indX0 = x
                if (x == grid.shape[0] - 1):
                    indX1 = x
                else:
                    indX1 = x + 1
            else:
                deltaX = (point[0] + cell_sizeX - cell_center_X) / cell_sizeX
                if (x == 0):
                    indX0 = x
                else:
                    indX0 = x - 1
                indX1 = x
            if (up):
                deltaY = (point[1] - cell_center_Y) / cell_sizeY
                indY0 = y
                if (y == grid.shape[1] - 1):
                    indY1 = y
                else:
                    indY1 = y + 1
            else:
                deltaY = (point[1] + cell_sizeY - cell_center_Y) / cell_sizeY
                if (y == 0):
                    indY0 = y
                else:
                    indY0 = y - 1
                indY1 = y
            result, density = bilinear_interpolation(grid[indX0,indY0], grid[indX1,indY0], grid[indX0,indY1], grid[indX1,indY1], deltaX, deltaY)
            return np.array([result[0] * cell_sizeX, result[1] * cell_sizeY]), density

        def Distance_component(point, prev_point, next_point):
            '''Ensures the point to respect the equal distance requirement of the method by
            maintaining equal distance with its neighbours. Tries to keep the point where it was'''
            if (prev_point == next_point):
                return 0
            u = np.array([next_point[0] - prev_point[0], next_point[1] - prev_point[1]])
            v = np.array([point[0] - prev_point[0], point[1] - prev_point[1]])
            center = u * (np.dot(u,v) / np.dot(u,u))
            center = [prev_point[0] + center[0], prev_point[1] + center[1]]
            m1 = np.array([prev_point[0] - center[0], prev_point[1] - center[1]])
            m2 = np.array([next_point[0] - center[0], next_point[1] - center[1]])
            return (m1+m2)/2.0

        def Angle_component(point, prev_point, next_point):
            '''Maximizes vertex angle and minimizes curvature in order
            to obtain less meandering trajectories'''
            n1 = np.array([prev_point[0] - point[0], prev_point[1] - point[1]])
            n2 = np.array([next_point[0] - point[0], next_point[1] - point[1]])
            if np.all(n1 + n2 == 0):
                return np.array([0,0])
            # Distance from the origin a.k.a. L2 norm
            len1 = np.linalg.norm(n1)
            len2 = np.linalg.norm(n2)
            # Normalize n1 and n2 a.k.a tranform them into unit vectors
            n1 = n1/len1
            n2 = n2/len2

            factor = np.cbrt(np.dot(n1,n2)) + 1
            if factor == 0:
                return np.array([0,0])
            minDist = min(len1, len2)

            return ((n1 + n2) / np.linalg.norm(n1 + n2)) * (minDist * factor)

        def Endpoints_Projection(point, line_endpoint1, line_endpoint2):
            '''Projects the point onto the line defined by the two endpoints'''
            x = np.array(point)
            u = np.array(line_endpoint1)
            v = np.array(line_endpoint2)

            n = v - u
            if (np.all(n == 0)):
                return point
            n /= np.linalg.norm(n, 2)

            result = u + n*np.dot(x - u, n)
            return [result[0], result[1]]

        #start_time = time.perf_counter()
        for track in tracks:
            if len(track) < 3:
                continue
            # Initialize density values to avoid recomputations
            densityTrack = 0
            Momentum_component = [np.array([0,0]) for point in range(len(track)-2)]
            Surface_components = [[]] * (len(track)-2)
            for i in range(1,len(track)-1):
                Surface_components[i-1], densityP = Surface_component(grid, cell_sizeX, cell_sizeY, track[i], minX, minY)
                densityTrack += densityP
            densityTrack /= len(track)-2
            prev_densityTrack = 0
            iteration = 0
            #start = time.perf_counter()
            while iteration < 4000 and not(abs(prev_densityTrack - densityTrack) < SlideEpsilon):
                iteration += 1
                prev_densityTrack = densityTrack
                for i in range(1, len(track)-1):
                    dv = Distance_component(track[i], track[i-1], track[i+1])
                    av = Angle_component(track[i], track[i-1], track[i+1])
                    Momentum_component[i-1] = 0.5 * Surface_components[i-1] + 0.2 * dv + 0.1 * av + 0.7 * Momentum_component[i-1]
                    track[i] = [track[i][0] + Momentum_component[i-1][0], track[i][1] + Momentum_component[i-1][1]]
                    Surface_components[i-1], densityP = Surface_component(grid, cell_sizeX, cell_sizeY, track[i], minX, minY)
                    densityTrack += densityP
                track[0] = Endpoints_Projection(track[0], track[1], track[2])
                track[len(track)-1] = Endpoints_Projection(track[len(track)-1], track[len(track)-3], track[len(track)-2])
                densityTrack = 0.8*(densityTrack/len(track)-2) + 0.2*prev_densityTrack
            #end = time.perf_counter()
        #end_time = time.perf_counter()
        return tracks

    Slide = SlideMethod_adapted(Dgrid, cell_size_lon, cell_size_lat, tracks)

    if saveSlide:
        with open(filename + '_Slide' + str(SlideEpsilon) + '.kml', 'w') as new_file:
            new_file.write(KML_creator(Slide).to_string(prettyprint=True))

    # points could have been moved outside the actual grid, especially the endpoints
    minX = minY = sys.float_info.max
    maxX = maxY = sys.float_info.min
    for track in Slide:
        for coord in track:
            if (coord[0] < minX):
                minX = coord[0]
            elif (coord[0] > maxX):
                maxX = coord[0]
            if (coord[1] < minY):
                minY = coord[1]
            elif (coord[1] > maxY):
                maxY = coord[1]

    Dgrid, cell_size_lon, cell_size_lat, freq_grid = density_grid(cell_size, Slide, True, minX, minY, maxX, maxY)

    binary_img = Dgrid > ImgFilter

    def Fill_surrounded_empty_cells(grid):
        '''Fills with a 1 the cells with 0s that are surrounded by 1s
        in their 4 nearest neighbour cells'''
        changed = True
        while changed:
            changed = False
            lengthX = grid.shape[0]
            lengthY = grid.shape[1]
            for x in range(lengthX):
                for y in range(lengthY):
                    if grid[x,y] == 0:
                        if x != 0 and grid[x-1,y] != 1:
                            break
                        if x != lengthX - 1 and grid[x+1,y] != 1:
                            break
                        if y != 0 and grid[x,y-1] != 1:
                            break
                        if y != lengthY - 1 and grid[x,y+1] != 1:
                            break
                        changed = True
                        grid[x,y] = 1
        return grid
                

    img = Fill_surrounded_empty_cells(binary_img)

    G123_LUT = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1,
           0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0,
           1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
           0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1,
           0, 0, 0], dtype=np.bool)

    G123P_LUT = np.array([0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
           1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0,
           0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0,
           1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1,
           0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0], dtype=np.bool)
    def bwmorph_thin(image, n_iter=None):
        '''Perform morphological thinning of a binary image'''
        # check parameters
        if n_iter is None:
            n = -1
        elif n_iter <= 0:
            raise ValueError('n_iter must be > 0')
        else:
            n = n_iter
        
        # check that we have a 2d binary image, and convert it
        # to uint8
        skel = np.array(image).astype(np.uint8)
        
        if skel.ndim != 2:
            raise ValueError('2D array required')
        if not np.all(np.in1d(image.flat,(0,1))):
            raise ValueError('Image contains values other than 0 and 1')

        # neighborhood mask
        mask = np.array([[ 8,  4,  2],
                         [16,  0,  1],
                         [32, 64,128]],dtype=np.uint8)

        # iterate either 1) indefinitely or 2) up to iteration limit
        while n != 0:
            before = np.sum(skel) # count points before thinning
            
            # for each subiteration
            for lut in [G123_LUT, G123P_LUT]:
                # correlate image with neighborhood mask
                N = ndi.correlate(skel, mask, mode='constant')
                # take deletion decision from this subiteration's LUT
                D = np.take(lut, N)
                # perform deletion
                skel[D] = 0
                
            after = np.sum(skel) # coint points after thinning
            
            if before == after:
                # iteration had no effect: finish
                break
                
            # count down to iteration limit (or endlessly negative)
            n -= 1
        
        return skel.astype(np.bool)

    thinned_img = bwmorph_thin(img)

    def Clean_isolated_points(grid):
        '''Removes the isolated 1s. That means, the 1s that are surraounded by
        eight 0s around their neighbour cells'''
        lengthX = grid.shape[0]
        lengthY = grid.shape[1]
        for x in range(lengthX):
            for y in range(lengthY):
                if grid[x,y] == 1:
                    if x != 0 and grid[x-1,y] != 0:
                        break
                    if (x != 0 and y != 0) and grid[x-1,y-1] != 0:
                        break
                    if (x != 0 and y != lengthY - 1) and grid[x-1,y+1] != 0:
                        break
                    if x != lengthX - 1 and grid[x+1,y] != 0:
                        break
                    if (x != lengthX - 1 and y != 0) and grid[x+1,y-1] != 0:
                        break
                    if (x != lengthX - 1 and y != lengthY - 1) and grid[x+1,y+1] != 0:
                        break
                    if y != 0 and grid[x,y-1]:
                        break
                    if y != lengthY - 1 and grid[x,y+1] != 0:
                        break
                    grid[x,y] = 0
        return grid

    thinned_img = Clean_isolated_points(thinned_img)
    thinned_img = thinned_img.astype(int)

    def Pixel_weights(grid, skeleton):
        '''Returns the frequency of each edge defined by two adjacent cells,
        which is the number of trajectories mapped to the skeleton pixel passing through it'''
        # Distance transform by indices to know the closest 1 of the skeleton img from each cell
        useless, distance_transform = ndi.distance_transform_edt(1 - skeleton, return_indices=True)
        result = np.zeros(skeleton.shape)
        lengthX = grid.shape[0]
        lengthY = grid.shape[1]
        for x in range(lengthX):
            for y in range(lengthY):
                result[distance_transform[0][x,y],distance_transform[1][x,y]] += grid[x,y]
        return result

    pixel_freq = Pixel_weights(freq_grid, thinned_img)

    def Tracks_from_binary_img(grid):
        '''Makes a DFS over all the grid to extract all the connected trajectories.
        (it repeats initial vertices if they have degree higher than 1 to produce a correct kml file)'''

        tracks = []
        new_track = []
        visited = np.full(grid.shape, False)
        stack = []
        lengthX = grid.shape[0]
        lengthY = grid.shape[1]
        for i in range(lengthX):
            for j in range(lengthY):
                if not visited[i,j]:
                    visited[i,j] = True
                    if grid[i,j]:
                        stack.append([[i,j],[i,j]])
                        index = len(tracks)
                        while stack:
                            point = stack.pop()
                            if not new_track:
                                new_track.append(point[1])
                            new_track.append(point[0])
                            added = False
                            x = point[0][0]
                            y = point[0][1]
                            if x != 0 and y != 0 and grid[x-1,y-1] and not visited[x-1,y-1]:
                                added = True
                                visited[x-1,y-1] = True
                                stack.append([[x-1,y-1],[x,y]])
                            if x != 0 and grid[x-1,y] and not visited[x-1,y]:
                                added = True
                                visited[x-1,y] = True
                                stack.append([[x-1,y],[x,y]])
                            if x != 0 and y != lengthY - 1 and grid[x-1,y+1] and not visited[x-1,y+1]:
                                added = True
                                visited[x-1,y+1] = True
                                stack.append([[x-1,y+1],[x,y]])
                            if y != lengthY - 1 and grid[x,y+1] and not visited[x,y+1]:
                                added = True
                                visited[x,y+1] = True
                                stack.append([[x,y+1],[x,y]])
                            if x != lengthX - 1 and y != lengthY - 1 and grid[x+1,y+1] and not visited[x+1,y+1]:
                                added = True
                                visited[x+1,y+1] = True
                                stack.append([[x+1,y+1],[x,y]])
                            if x != lengthX - 1 and grid[x+1,y] and not visited[x+1,y]:
                                added = True
                                visited[x+1,y] = True
                                stack.append([[x+1,y],[x,y]])
                            if x != lengthX - 1 and y != 0 and grid[x+1,y-1] and not visited[x+1,y-1]:
                                added = True
                                visited[x+1,y-1] = True
                                stack.append([[x+1,y-1],[x,y]])
                            if y != 0 and grid[x,y-1] and not visited[x,y-1]:
                                added = True
                                visited[x,y-1] = True
                                stack.append([[x,y-1],[x,y]])

                            if not added:
                                tracks.append([item for item in new_track])
                                new_track = []
                        tracks[index].pop(0)
        return tracks

    tracks_vertices = Tracks_from_binary_img(thinned_img)

    def Simplify_tracks(tracks, frequencies):
        '''The simplification consists of applying Ramer-Douglas-Peucker algorithm
        to reduce the number of vertices of each trajectory, and, later delete the
        edges weights associated with the removed vertices. The edges weights are updated
        with respect their new endpoints (length and the maximum freq among the removed
        intermediate vertices). It is also computed the maximum frequency among all the
        vertices of the result'''
        result = []
        weights = []
        lengthList = []
        min_freq = sys.float_info.max
        for i, track in enumerate(tracks):
            # the epsilon is due to the fact that the vertices are in cell indexes
            result.append(rdp(track, epsilon=2/cell_size))

            edges_weights = []
            index = 1
            # initialize with the first vertex weight
            max_freq = frequencies[track[0][0],track[0][1]]
            for j in range(1,len(track)):
                # We take the maximum weight among all the vertices that were the new edge
                if max_freq < frequencies[track[j][0],track[j][1]]:
                    max_freq = frequencies[track[j][0],track[j][1]]
                if min_freq > frequencies[track[j][0],track[j][1]]:
                    min_freq = frequencies[track[j][0],track[j][1]]

                if track[j] == result[i][index]:
                    # We have found the endpoint of the edge
                    edge_length = ((((result[i][index-1][0] - result[i][index][0])**2) + ((result[i][index-1][1] - result[i][index][1])**2))**0.5)
                    edges_weights.append([edge_length, max_freq])
                    lengthList.append(edge_length)
                    if j != len(track)-1:
                        max_freq = frequencies[track[j+1][0],track[j+1][1]] # we get the next one a priory to initialize correctly
                    index += 1
            weights.append(edges_weights)

        return result, weights, lengthList, min_freq

    simplified_tracks, edges_weights, lengths, frequency_threshold = Simplify_tracks(tracks_vertices, pixel_freq)

    lengths.sort()
    length_threshold = lengths[len(lengths)//2] # median

    def Delete_underweighted_edges(tracks, weights, frequency, length):
        '''Deletes edges and the correspondent vertices if they have smaller
        frequency and length than the thresholds'''
        result_tracks = []
        result_weights = []
        
        for t, track_edges in enumerate(weights):
            new_vertices = []
            new_edges = []
            deleted = True
            for i in range(len(track_edges)):
                if track_edges[i][0] >= length and track_edges[i][1] > frequency:
                    deleted = False
                    new_vertices.append(tracks[t][i])
                    new_edges.append(track_edges[i])
                elif not deleted:
                    deleted = True
                    new_vertices.append(tracks[t][i])
                    result_tracks.append([vertex for vertex in new_vertices])
                    result_weights.append([edge for edge in new_edges])
                    new_vertices = []
                    new_edges = []
            if not deleted:
                new_vertices.append(tracks[t][len(tracks[t])-1])
                result_tracks.append(new_vertices)
                result_weights.append(new_edges)
        return result_tracks, result_weights

    if delete_edges:
        vertices, edges = Delete_underweighted_edges(simplified_tracks, edges_weights, frequency_threshold, length_threshold)
    else:
        vertices = simplified_tracks
        edges = edges_weights

    def Plot_all_tracks(tracks, weights):
        '''Plots the tracks with the edge weights as colors. From yellow to red increasing weight'''

        def colorGradient(weight, max_weight):
            '''Returns a color from yellow to red depending on the weight'''
            return (1,1-(weight/max_weight),0)

        #First we need to determine the max frequency weight for the gradient
        max_freq = -1
        for track_edges in weights:
            for edge in track_edges:
                if edge[1] > max_freq:
                    max_freq = edge[1]

        for t, track in enumerate(tracks):
            x = [coord[0] for coord in track]
            y = [coord[1] for coord in track]
            for i in range(len(track)-1):
                plt.plot(x[i:i+2], y[i:i+2], '-', color=colorGradient(weights[t][i][1], max_freq))
        plt.axis('equal')
        plt.show()

    if plot_result:
        Plot_all_tracks(vertices, edges)

    #before returning we must return the vertices to its original coordinate system
    result = []
    for track in vertices:
        new_track = []
        for point in track:
            new_track.append([minX + cell_size_lon * (point[0] + 0.5), minY + cell_size_lat * (point[1] + 0.5)])
        result.append(new_track)

    return result, edges

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('-i', '--input_file', default='delta.kmz', help='The input file containing the KML data')
    argparser.add_argument('-n', '--filename', default='road_network', type=str, help='The input file containing the KML data')
    argparser.add_argument('-c', '--cell', default=3, type=float, help='Number of meters of a grid cell')
    argparser.add_argument('-r', '--resample_meters', default=10, type=float, help='Number of meters between points of tracks')
    argparser.add_argument('-s', '--sigma', default=3, type=float, help='Sigma for gaussian blurs')
    argparser.add_argument('-f', '--filter', default=0.05, type=float, help='Density threshold to obatin the binary image filtering the density grid')
    argparser.add_argument('-e', '--slide', default=0.0001, type=float, help='Epislon to stop iterating a track in Slide method')
    argparser.add_argument('-eq', '--equidistance_plots', default=False, type=bool, help='Shows all the plot results of all the tracks for the equidistance resample')
    argparser.add_argument('-pl', '--plot', default=False, type=bool, help='Plots the final result')
    argparser.add_argument('-ed', '--edges', default=False, type=bool, help='Delete underweighted edges')
    argparser.add_argument('-ss', '--slidekml', default=False, type=bool, help='Saves a kml with the Slide result')
    args = argparser.parse_args()

    # Read input_file
    kmz = ZipFile('data/tracks.kmz', 'r')
    file = kmz.open('doc.kml', 'r').read()
    data = KmlTrackReader(file)
    kmz.close()
    # Apply the map construction algorithm
    result_vertices, result_edges = Road_Network(data.tracks, args.filename, args.cell, args.resample_meters, args.equidistance_plots, args.plot, args.sigma, args.slide, args.filter, args.edges, args.slidekml)
    # Save the result in a kml file
    with open(args.filename + str(args.slide) + '.kml', 'w') as new_file:
        new_file.write(KML_creator(result_vertices).to_string(prettyprint=True))