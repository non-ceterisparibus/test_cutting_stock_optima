from gurobipy import *

number_of_vehicles = 2 
vehicles_origin = [4, 6]
vehicles_destination = [3, 0]
total_time_vehicle = [35, 27]

truck = [i for i in range(number_of_vehicles)]
starting_node = {}
destination_nodes = {}
time = {}
for i in truck:
  starting_node[i] = vehicles_origin[i]
  destination_nodes[i] = vehicles_destination[i]
  time[i] = total_time_vehicle[i]


multi = {}
for i in range(number_of_vehicles):
  l = [vehicles_origin[i],vehicles_destination[i],total_time_vehicle[i]]
  multi[i] = l

truck, starting_node, destination_nodes, time = multidict(multi)