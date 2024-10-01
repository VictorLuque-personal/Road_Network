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
from Road_network import Road_Network
from enum import Enum
import copy
from sympy import Point, Line, Segment

argparser = argparse.ArgumentParser()
argparser.add_argument('-i', '--input_file', default='delta.kmz', help='The input file containing the KML data')
argparser.add_argument('-n', '--filename', default='splitANDmerge', type=str, help='The input file containing the KML data')
argparser.add_argument('-c', '--cell', default=3, type=float, help='Number of meters of a grid cell')
argparser.add_argument('-r', '--resample_meters', default=10, type=float, help='Number of meters between points of tracks')
argparser.add_argument('-s', '--sigma', default=3, type=float, help='Sigma for gaussian blurs')
argparser.add_argument('-f', '--filter', default=0.05, type=float, help='Density threshold to obatin the binary image filtering the density grid')
argparser.add_argument('-e', '--slide', default=0.0001, type=float, help='Epislon to stop iterating a track in Slide method')
argparser.add_argument('-eq', '--equidistance_plots', default=False, type=bool, help='Shows all the plot results of all the tracks for the equidistance resample')
argparser.add_argument('-pl', '--plot', default=False, type=bool, help='Plots the final result')
argparser.add_argument('-ed', '--edges', default=False, type=bool, help='Delete underweighted edges')
argparser.add_argument('-o', '--overlap', default=10, type=float, help='Number of cells to add for overlapping zones')
argparser.add_argument('-p', '--partitions', default=4, type=int, help='Number of partitions for the Split&Merge')
argparser.add_argument('-m', '--merge_distance', default=3*3*3, type=float, help='Distance threshold in number of cells to determine which tracks to merge')
args = argparser.parse_args()

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


# Read input_file
kmz = ZipFile('data/' + args.input_file, 'r')
file = kmz.open('doc.kml', 'r').read()
data = KmlTrackReader(file)
kmz.close()

minX = minY = sys.float_info.max
maxX = maxY = sys.float_info.min
for track in data.tracks:
    for coord in track:
        if (coord[0] < minX):
            minX = coord[0]
        elif (coord[0] > maxX):
            maxX = coord[0]
        if (coord[1] < minY):
            minY = coord[1]
        elif (coord[1] > maxY):
            maxY = coord[1]

def meters_to_latitude(meters):
    '''A nautical mile is 1852 meters, that is one minute of arc at the meridian.
    So, in degrees there are 1852 * 60 = 111.120 aprox'''
    return meters / 111120

def meters_to_longitude(meters, latitude):
    '''Depending on latitude. Along the equator there are aprox 40074784 meters.
    So, one degree of longitude at the equator is 111319 aprox'''
    return meters * abs(math.cos(latitude)) / 111319

def Distance_between_points(point1, point2):
    return ((((point2[0] - point1[0])**2) + ((point2[1] - point1[1])**2) )**0.5)

class Direction(Enum):
    UP = 0
    DOWN = 1
    RIGHT = 2
    LEFT = 3

def Split_Quadrants(tracks, number_partitions, minX, minY, maxX, maxY):
    '''Splits the area that cover all the tracks in {number_partitions} parts'''

    def Split_Quadrant(quadrant, axisX, key, overlaps):
        '''Splits a quadrant defined by two points in half. Returns a list with the new quadrants'''
        new_quadrant1 = []
        new_quadrant2 = []
        # Split in half by the X axis, thus, a right and a left half quadrants
        if axisX:
            new_quadrant1 = [quadrant[1][0], [(quadrant[1][0][0] + quadrant[1][1][0])/2, quadrant[1][1][1]]] # left half
            new_quadrant2 = [[(quadrant[1][0][0] + quadrant[1][1][0])/2, quadrant[1][0][1]], quadrant[1][1]] # right half
            # Now update the others if they exist or create new ones
            overlaps[str(key) + Direction.LEFT.name] = quadrant[0] # this one is new
            # Right update for the quadrant and its right quadrant (if it already exists)
            if str(quadrant[0]) + Direction.RIGHT.name in overlaps:
                aux = overlaps[str(quadrant[0]) + Direction.RIGHT.name]
                overlaps[str(key) + Direction.RIGHT.name] = aux
                overlaps[str(aux) + Direction.LEFT.name] = key
            overlaps[str(quadrant[0]) + Direction.RIGHT.name] = key
            # Up update for the quadrant and its upper quadrant or quadrants (if they already exist)
            if str(quadrant[0]) + Direction.UP.name in overlaps:
                aux = overlaps[str(quadrant[0]) + Direction.UP.name]
                if isinstance(aux, dict):
                    overlaps[str(key) + Direction.UP.name] = aux["right"]
                    overlaps[str(aux["right"]) + Direction.DOWN.name] = key
                    overlaps[str(quadrant[0]) + Direction.UP.name] = aux["left"]
                else:
                    overlaps[str(aux) + Direction.DOWN.name] = { "left": quadrant[0], "right": key }
                    overlaps[str(key) + Direction.UP.name] = aux

            # Down update for the quadrant and its lower quadrant or quadrants (if they already exist)
            if str(quadrant[0]) + Direction.DOWN.name in overlaps:
                aux = overlaps[str(quadrant[0]) + Direction.DOWN.name]
                if isinstance(aux, dict):
                    overlaps[str(key) + Direction.DOWN.name] = aux["right"]
                    overlaps[str(aux["right"]) + Direction.UP.name] = key
                    overlaps[str(quadrant[0]) + Direction.DOWN.name] = aux["left"]
                else:
                    overlaps[str(aux) + Direction.UP.name] = { "left": quadrant[0], "right": key }
                    overlaps[str(key) + Direction.DOWN.name] = aux
        # Split in half by the Y axis, thus, an upper and a lower half quadrants
        else:
            new_quadrant1 = [quadrant[1][0], [quadrant[1][1][0], (quadrant[1][0][1] + quadrant[1][1][1])/2]] # lower half
            new_quadrant2 = [[quadrant[1][0][0], (quadrant[1][0][1] + quadrant[1][1][1])/2], quadrant[1][1]] # upper half
            # Now update the others if they exist or create new ones
            overlaps[str(key) + Direction.DOWN.name] = quadrant[0] # this one is new
            # Up update for the quadrant and its upper quadrant (if it already exists)
            if str(quadrant[0]) + Direction.UP.name in overlaps:
                aux = overlaps[str(quadrant[0]) + Direction.UP.name]
                overlaps[str(key) + Direction.UP.name] = aux
                overlaps[str(aux) + Direction.DOWN.name] = key
            overlaps[str(quadrant[0]) + Direction.UP.name] = key
            # Right update for the quadrant and its right quadrant or quadrants (if they already exist)
            if str(quadrant[0]) + Direction.RIGHT.name in overlaps:
                aux = overlaps[str(quadrant[0]) + Direction.RIGHT.name]
                if isinstance(aux, dict):
                    overlaps[str(key) + Direction.RIGHT.name] = aux["up"]
                    overlaps[str(aux["up"]) + Direction.LEFT.name] = key
                    overlaps[str(quadrant[0]) + Direction.RIGHT.name] = aux["down"]
                else:
                    overlaps[str(aux) + Direction.LEFT.name] = { "down": quadrant[0], "up": key }
                    overlaps[str(key) + Direction.RIGHT.name] = aux

            # Left update for the quadrant and its left quadrant or quadrants (if they already exist)
            if str(quadrant[0]) + Direction.LEFT.name in overlaps:
                aux = overlaps[str(quadrant[0]) + Direction.LEFT.name]
                if isinstance(aux, dict):
                    overlaps[str(key) + Direction.LEFT.name] = aux["up"]
                    overlaps[str(aux["up"]) + Direction.RIGHT.name] = key
                    overlaps[str(quadrant[0]) + Direction.LEFT.name] = aux["down"]
                else:
                    overlaps[str(aux) + Direction.RIGHT.name] = { "down": quadrant[0], "up": key }
                    overlaps[str(key) + Direction.LEFT.name] = aux
                    
        return { quadrant[0]: new_quadrant1, key: new_quadrant2 }, overlaps

    def Add_offset(quadrant, cell_sizeX, cell_sizeY):
        '''Adds an offset to the quadrant to abtain an overlapping
        zone for the algorithm'''
        quadrant[1][0][0] -= args.overlap * cell_sizeX
        quadrant[1][0][1] -= args.overlap * cell_sizeY
        quadrant[1][1][0] += args.overlap * cell_sizeX
        quadrant[1][1][1] += args.overlap * cell_sizeY
        return quadrant

    keys = 1
    iterations = number_partitions
    dict1 = { 0 : [[minX, minY], [maxX, maxY]] }
    dict2 = {}
    axisX = True
    overlaps = {}

    while iterations > 1: # to get x partitions, x-1 splits are needed
        iterations -= 1
        new_quadrants, overlaps = Split_Quadrant(dict1.popitem(), axisX, keys, overlaps)
        dict2.update(new_quadrants)
        keys += 1
        if not dict1:
            axisX = not axisX
            dict1 = dict2.copy()
            dict2 = {}

    dict1.update(dict2)

    cell_size_lat = meters_to_latitude(args.cell)
    cell_size_lon = meters_to_longitude(args.cell, (minY + maxY) / 2)

    for quadrant in dict1.items():
        quadrant = Add_offset(quadrant, cell_size_lon, cell_size_lat)

    return dict1, overlaps

quadrants, overlaps = Split_Quadrants(data.tracks, args.partitions, minX, minY, maxX, maxY)

def line_intersection(line1, line2):
    s1 = Segment(line1[0], line1[1])
    s2 = Segment(line2[0], line2[1])
      
    # using intersection() method
    intersection = s1.intersection(s2)
    if intersection:
        return [float(item) for item in list(intersection[0])]
    else:
        return intersection

def Split_data(tracks, quadrants):
    '''Splits the input data according to the quadrants. It returns
    a list with the datasets of tracks for each quadrant'''

    def quadrant_segment(quadrant, line):
        '''Depending on the parameter returns one of the four segments that define a quadrant'''
        if line == 0:
            return [[quadrant[0][0], quadrant[0][1]], [quadrant[1][0], quadrant[0][1]]]
        if line == 1:
            return [[quadrant[1][0], quadrant[0][1]], [quadrant[1][0], quadrant[1][1]]]
        if line == 2:
            return [[quadrant[1][0], quadrant[1][1]], [quadrant[0][0], quadrant[1][1]]]
        return [[quadrant[0][0], quadrant[1][1]], [quadrant[0][0], quadrant[0][1]]]

    result = {}
    new_dataset = []
    new_track = []
    if len(quadrants) == 1:
        result[0] = tracks
        return result
    for i, quadrant in quadrants.items():
        for track in tracks:
            outside = True
            for p, point in enumerate(track):
                if point[0] >= quadrant[0][0] and point[0] <= quadrant[1][0] and point[1] >= quadrant[0][1] and point[1] <= quadrant[1][1]:
                    if outside and p != 0:
                        for j in range(4):
                            seg1 = quadrant_segment(quadrant, j)
                            seg2 = [track[p-1], point]
                            if line_intersection(seg1, seg2):
                                new_track.append(line_intersection(seg1, seg2))
                                break
                    outside = False
                    new_track.append(point)
                elif new_track:
                    # first append an extra vertex at the border to preserve the original track
                    for j in range(4):
                        seg1 = quadrant_segment(quadrant, j)
                        seg2 = [new_track[len(new_track)-1], point]
                        if line_intersection(seg1, seg2):
                            new_track.append(line_intersection(seg1, seg2))
                            break
                    new_dataset.append([point for point in new_track])
                    new_track = []
                    outside = True
                else:
                    outside = True
            if new_track:
                new_dataset.append([point for point in new_track])
                new_track = []
        result[i] = [track for track in new_dataset]
        new_dataset = []
    return result, quadrant_times

datasets, times = Split_data(data.tracks, quadrants)

def Plot_all_tracks(tracks):
    for track in tracks:
        x = [coord[0] for coord in track]
        y = [coord[1] for coord in track]
        plt.plot(x, y, '-r')
    plt.axis('equal')
    plt.show()

for k, dataset in datasets.items():
    print(k)
    Plot_all_tracks(dataset)

results = {}
edges = {}
for i, dataset in datasets.items():
    results[i], edges[i] = Road_Network(dataset, '', args.cell, args.resample_meters, args.equidistance_plots, args.plot, args.sigma, args.slide/args.partitions, args.filter, args.edges, False)
del edges

# now we have to merge the results properly
def Merge(datasets, overlaps, distance_threshold_X, distance_threshold_Y, splits):
    '''To merge all quadrants we have overlapping zones between contiguous
    quadrants. By finding the intersections of both contiguous quadrants
    with the line in the middle of the overlapping zone, we can merge nearby
    tracks from different quadrants'''

    def Merge_Row(quadrants, datasets, distance_threshold_X, distance_threshold_Y, splits):
        '''Merges the quadrants of a row. The clue is to compute first the cells
        that are one over the other and later do the merge incrementally by side'''

        iterator = 1
        result = datasets[quadrants[0]]
        result_split = splits[quadrants[0]]
        while iterator <= len(quadrants) - 1:
            if isinstance(quadrants[iterator], dict):
                # if the right part is splitted, first we have to merge it and then merge with the left one
                line, aux_split = Splitting_line(splits[quadrants[iterator]['down']], splits[quadrants[iterator]['up']], False)
                new_dataset = Basic_Merge(datasets[quadrants[iterator]['down']], datasets[quadrants[iterator]['up']], distance_threshold_Y, line, False)
                del datasets[quadrants[iterator]['down']]
                del datasets[quadrants[iterator]['up']]
                line, result_split = Splitting_line(result_split, aux_split, True)
                result = Basic_Merge(result, new_dataset, distance_threshold_X, line, True)
            else:
                line, result_split = Splitting_line(result_split, splits[quadrants[iterator]], True)
                result = Basic_Merge(result, datasets[quadrants[iterator]], distance_threshold_X, line, True)
            iterator += 1
        return result, result_split

    def Basic_Merge(first_quadrant, second_quadrant, distance_threshold, splitting_line, axisX):
        '''Merges two contiguous quadrants'''
        
        if axisX:
            first_intersections_new = Find_intersections(first_quadrant, splitting_line, Direction.RIGHT)
            second_intersections_new = Find_intersections(second_quadrant, splitting_line, Direction.LEFT)
        else:
            first_intersections_new = Find_intersections(first_quadrant, splitting_line, Direction.UP)
            second_intersections_new = Find_intersections(second_quadrant, splitting_line, Direction.DOWN)
        
        result = []
        for key in first_intersections_new:
            finish = len(first_intersections_new[key])
            i = 0
            while i < finish:
                new_track, first_intersections_new, second_intersections_new = Merge_intersection(key, i, first_intersections_new, second_intersections_new, first_quadrant, second_quadrant, distance_threshold)
                if new_track:
                    result.append(new_track) # This will be the new tracks
                    finish = len(first_intersections_new[key])
                else:
                    i += 1
        # this is unefficient but avoid memory fullness, so in some way it is more efficient
        if axisX:
            first_intersections = Find_intersections(first_quadrant, splitting_line, Direction.RIGHT)
            second_intersections = Find_intersections(second_quadrant, splitting_line, Direction.LEFT)
        else:
            first_intersections = Find_intersections(first_quadrant, splitting_line, Direction.UP)
            second_intersections = Find_intersections(second_quadrant, splitting_line, Direction.DOWN)
        # Now we have to remove properly the parts of the tracks that have changed
        return Merged_quadrant(first_intersections_new, first_intersections, second_intersections_new, second_intersections, result, first_quadrant, second_quadrant)

    def Merge_intersection(key, i, first_intersections, second_intersections, first_quadrant, second_quadrant, distance_threshold):
        '''Merge a track until its new end between both quadrants'''

        def concatenate_track(intersections, key, i, j, new_track, value, new_intersections, quadrant, second_quadrant):
            '''Finds the next intersection if it exists and concatenates the next part of the track'''
            
            # we try to find next intersection
            for key2, value2 in intersections.items():
                for k, intersection2 in enumerate(value2):
                    if Distance_between_points(value[i][1], intersection2[1]) <= distance_threshold:
                        # Right now we have found an intersection to continue within the other quadrant
                        if value[i][2]:
                            # the first part of the track until the intersection is the one we want
                            new_track += quadrant[key][j+1:value[i][0]+1] # get the first part
                        else:
                            # the last part of the track until the intersection is the one we want
                            aux = []
                            if j == 0 or j == -1:
                                aux = quadrant[key][value[i][0]+1:] # get the last part
                                aux.reverse() # to append at the end
                            else:
                                aux = quadrant[key][value[i][0]+1:j+1] # get the last part
                                aux.reverse() # to append at the end
                            new_track += aux
                        # compute the midpoint between the two involved in the intersection
                        new_track.append([(value[i][1][0] + intersection2[1][0])/2., (value[i][1][1] + intersection2[1][1])/2.])
                        # Now we have to initialize the values to continue the main function loop
                        value.pop(i)
                        new_intersections[key] = value
                        j = intersection2[0]
                        if intersection2[2]:
                            if k == 0:
                                # there are no more intersections
                                aux = second_quadrant[key2][:intersection2[0]+1] # get the first part
                                aux.reverse() # to append at the end
                                new_track += aux
                                value2.pop(k)
                                intersections[key2] = value2
                                key = key2
                                value = value2
                                # Finish (the rest of the return values will be ignored)
                                return True, new_track, key, i, j, value, new_intersections, intersections
                            else:
                                i = k - 1
                        else:
                            if k == len(value2)-1:
                                # there are no more intersections
                                aux = second_quadrant[key2][intersection2[0]+1:] # get the last part
                                new_track += aux
                                value2.pop(k)
                                intersections[key2] = value2
                                key = key2
                                value = value2
                                return True, new_track, key, i, j, value, new_intersections, intersections # Finish
                        value2.pop(k)
                        key = key2
                        value = value2
                        intersections[key2] = value2
                        return False, new_track, key, i, j, value, new_intersections, intersections
            # There is no itnersecion for this point. We finish 
            return True, new_track, key, i, j, value, new_intersections, intersections
        
        # initialization
        get_second = True # to know which quadrant we are merging with the other one
        j = -1 # position where strats/ends the part of the track to concatenate
        finish = False
        new_track = []
        value = first_intersections[key]
        while not finish:
            finish, new_track, key, i, j, value, new_intersections1, new_intersections2 = concatenate_track(second_intersections if get_second else first_intersections,
                key, i, j, new_track, value, second_intersections if not get_second else first_intersections,
                first_quadrant if get_second else second_quadrant, first_quadrant if not get_second else second_quadrant)
            if get_second:
                first_intersections = new_intersections1
                second_intersections = new_intersections2
            else:
                second_intersections = new_intersections1
                first_intersections = new_intersections2
            get_second = not get_second
        return new_track, first_intersections, second_intersections

    def Merged_quadrant(first_intersections_new, first_intersections, second_intersections_new, second_intersections, new_tracks, first_quadrant, second_quadrant):
        '''Creates a new quadrant with the information of the tracks to remove/edit/add.
        We have the new tracks and we know the intersections that have been processed.
        The ones that not, and the tracks that had no intersections must remain equal'''
        
        result = []
        for i in range(len(first_quadrant)):
            if not (i in first_intersections):
                # the track remains equal, so we put it in the solution
                result.append(first_quadrant[i])
            else:
                # the track changed
                j = 0
                new_track = []
                last_intersection = None
                while j < len(first_intersections[i]):
                    if first_intersections[i][j] in first_intersections_new[i]:
                        # this means that the intersection was not merged with any other
                        if not last_intersection:
                            # right now, the direction is no longer important because these tracks are independent
                            if j == len(first_intersections[i])-1: # if it is the last intersection we take all the track
                                new_track = first_quadrant[i]
                                break
                            new_track = first_quadrant[i][:first_intersections[i][j+1][0]+1]
                            last_intersection = j+1
                        else:
                            if last_intersection != j and new_track:
                                result.append([point for point in new_track])
                                new_track = []
                            if j == len(first_intersections[i])-1: # if it is the last intersection we take all the rest
                                new_track += first_quadrant[i][first_intersections[i][j][0]+1:]
                            else:
                                new_track += first_quadrant[i][first_intersections[i][j][0]+1:first_intersections[i][j+1][0]+1]
                            # the one that have been cut off may repeat some vertices. Nevertheless this will almost never happen
                    j += 1
                if new_track:
                    result.append([point for point in new_track])
                    new_track = []


        # Now we have to maintain the tracks that haven't changed from the other quadrant
        for i in range(len(second_quadrant)):
            if not (i in second_intersections):
                # the track remains equal, so we put it in the solution
                result.append(second_quadrant[i])
            else:
                # the track changed
                j = 0
                new_track = []
                last_intersection = None
                while j < len(second_intersections[i]):
                    if second_intersections[i][j] in second_intersections_new[i]:
                        # this means that the intersection was not merged with any other
                        if not last_intersection:
                            # right now, the direction is no longer important because these tracks are independent
                            if j == len(second_intersections[i])-1: # if it is the last intersection we take all the track
                                new_track = second_quadrant[i]
                                break
                            new_track = second_quadrant[i][:second_intersections[i][j+1][0]+1]
                            last_intersection = j+1
                        else:
                            if last_intersection != j and new_track:
                                result.append([point for point in new_track])
                                new_track = []
                            if j == len(second_intersections[i])-1: # if it is the last intersection we take all the rest
                                new_track += second_quadrant[i][second_intersections[i][j][0]+1:]
                                break
                            new_track += second_quadrant[i][second_intersections[i][j][0]+1:second_intersections[i][j+1][0]+1]
                            # the one that have been cut off may repeat some vertices. Nevertheless this will almost never happen
                    j += 1
                if new_track:
                    result.append([point for point in new_track])
                    new_track = []
        result += new_tracks # Now that we have the independent tracks that did not merge, we must append the merged ones
        return result

    def Splitting_line(split1, split2, axisX):
        '''Find the splitting line between two splits depending on the direction.
        For horizontal splits, the first is the left one and the second the right one.
        For vertical splits, the first is the lower one and the second the upper one.'''
        if axisX:
            line = [[(split2[0][0] + split1[1][0])/2., split1[1][1]], [(split2[0][0] + split1[1][0])/2., split1[0][1]]]
        else:
            line = [[split1[0][0], (split2[0][1] + split1[1][1])/2.], [split1[1][0], (split2[0][1] + split1[1][1])/2.]]
        return line, [split1[0], split2[1]]


    def Row_quadrants(key, overlaps):
        '''Returns all the quadrants involved in a row'''
        # a priory we know that the partitions will never have a quadrant bigger than them at their right
        result = []
        result.append(key)
        while str(key) + Direction.RIGHT.name in overlaps:
            aux = overlaps[str(key) + Direction.RIGHT.name] # this could be a dictionary
            if isinstance(result[len(result)-1], dict):
                # we have to keep the row with the size of the first cell (at least in longitude)
                result.append({ 'up': aux, 'down': overlaps[str(result[len(result)-1]['down']) + Direction.RIGHT.name] })
            else:
                result.append(aux)
            
            if isinstance(aux, dict):
                key = aux['up'] # could be 'down'
            else:
                key = aux
        return result

    def Find_intersections(quadrant, splitting_line, direction):
        '''Returns where to find inside the quadrant, all the intersections
        between the splitting line and the trajectories of the quadrant'''
        def comparison(splitting_line_endpoint, direction, point):
            '''Returns de result of the correct comparison according to the splitting
            line and the direction where the quadrant overlaps with another'''
            if direction == Direction.UP:
                return splitting_line_endpoint[1] < point[1]
            if direction == Direction.DOWN:
                return splitting_line_endpoint[1] > point[1]
            if direction == Direction.LEFT:
                return splitting_line_endpoint[0] > point[0]
            if direction == Direction.RIGHT:
                return splitting_line_endpoint[0] < point[0]
        result = {}
        for i, track in enumerate(quadrant):
            for j, point in enumerate(track):
                # to avoid computations of intersections where it is impossible to find one
                if comparison(splitting_line[0], direction, point):
                    # Now we have a point that could be involved in an intersection
                    if not j == 0 and line_intersection(splitting_line, [track[j-1], point]):
                        if i in result:
                            result[i].append([j-1, line_intersection(splitting_line, [track[j-1], point]), True])
                        else:
                            result[i] = [[j-1, line_intersection(splitting_line, [track[j-1], point]), True]]
                    elif not j == len(track)-1 and line_intersection(splitting_line, [point, track[j+1]]):
                        if i in result:
                            result[i].append([j, line_intersection(splitting_line, [point, track[j+1]]), False])
                        else:
                            result[i] = [[j, line_intersection(splitting_line, [point, track[j+1]]), False]]
        return result

    # the clue is that we know that quadrant 0 is always the one at the lower left corner
    key = 0
    # first we are going to merge by rows, thus, contiguous by side (left, right) quadrants
    rows = []
    finish = False
    while not finish:
        rows.append(Merge_Row(Row_quadrants(key, overlaps), datasets, distance_threshold_X, distance_threshold_Y, splits))
        if str(key) + Direction.UP.name in overlaps:
            aux = overlaps[str(key) + Direction.UP.name]
            if isinstance(aux, dict):
                key = aux['left']
            else:
                key = aux
        else:
            finish = True
    
    result, result_split = rows.pop(0)
    i = 0
    while rows:
        next_quadrant, split2 = rows.pop(0)
        splitting_line, result_split = Splitting_line(result_split, split2, False)
        result = Basic_Merge(result, next_quadrant, distance_threshold_Y, splitting_line, False)
    return result

result = Merge(results, overlaps, meters_to_longitude(args.merge_distance, (minY + maxY) / 2), meters_to_latitude(args.merge_distance), quadrants)

def Plot_all_tracks(tracks):
    for track in tracks:
        x = [coord[0] for coord in track]
        y = [coord[1] for coord in track]
        plt.plot(x, y, '-r')
    plt.axis('equal')
    plt.show()

Plot_all_tracks(result)

cell_size_lat = meters_to_latitude(args.cell)
cell_size_lon = meters_to_longitude(args.cell, (minY + maxY) / 2)

def KML_creator(tracks):
    '''Creates a KML with the information of the parameter. It must be the dataset
    to write the trajectories'''
    result = kml.KML()
    ns = '{http://www.opengis.net/kml/2.2}'
    d = kml.Document(ns, '0', 'split&merge result')
    f = kml.Folder(name='TRACKS')
    result.append(d)
    d.append(f)
    for i, track in enumerate(tracks):
        placemark = kml.Placemark(ns, str(i), str(i))
        style = styles.Style(styles=[styles.LineStyle(color='ff0000ff'), styles.PolyStyle(fill=0)])
        placemark.append_style(style)
        trajectory = []
        for point in track:
            point = [minX + cell_size_lon * (point[0] + 0.5), minY + cell_size_lat * (point[1] + 0.5)]
            trajectory.append(tuple(point))
        placemark.geometry = LineString(trajectory)
        f.append(placemark)
    return result

# Saves the result in a kml file
with open('split&merge' + args.filename + str(args.partitions) + '.kml', 'w') as new_file:
    new_file.write(KML_creator(result).to_string(prettyprint=True))