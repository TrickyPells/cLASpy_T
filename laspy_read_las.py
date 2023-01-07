import numpy as np
import pandas as pd
import laspy
import csv

#chemin = "C:/Users/Xavier/Venvs/claspyT/"
chemin = "./Test/Orne_20130525.las"

las = laspy.read(chemin)
#las = laspy.open(chemin, mode='r')
# liste de toutes les dimensions
print(list(las.header.point_format.dimensions))

# liste de toutes les dimensions standard
print(list(las.header.point_format.standard_dimension_names))

# list de toutes les dimensions extra
print(list(las.header.point_format.extra_dimension_names))

# tous les points vers un np array
point_records = las.points
points_extradim = las.points[list(las.point_format.extra_dimension_names)]
points_target = las.points[['Target']]

# tous les points vers un dataframe
frame = pd.DataFrame(las.points.array)
frame_extra = pd.DataFrame(points_extradim.array)
frame_target = pd.DataFrame(points_target.array)
print(frame_target)

with open("./Test/Orne_20130525.csv", 'r') as file:
    #csvfile = csv.reader(file)
    print("row_count : ", sum(1 for line in file)-1)

print("Done")



