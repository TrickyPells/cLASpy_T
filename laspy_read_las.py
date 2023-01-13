import numpy as np
import pandas as pd
import laspy
import csv

#chemin = "C:/Users/Xavier/Venvs/claspyT/"
chemin = "./Test/Orne_20130525.las"

las = laspy.read(chemin)
#las = laspy.open(chemin, mode='r')
# liste de toutes les dimensions
print(list(las.header.point_format.dimension_names))

# liste de toutes les dimensions standard
print(list(las.header.point_format.standard_dimension_names))
print(list(las.point_format.standard_dimension_names))

# list de toutes les dimensions extra
print(list(las.header.point_format.extra_dimension_names))
print(list(las.point_format.extra_dimension_names))

# tous les points vers un np array
point_records = las.points
try:
    point_stddim = las.points[list(las.point_format.standard_dimension_names)]
except KeyError:
    pass

points_extradim = las.points[list(las.point_format.extra_dimension_names)]
points_target = las.points[['Target']]

# tous les points vers un dataframe
dtype_dict = dict()
for field in las.point_format.extra_dimension_names:
    dtype_dict[field] = np.float32

frame = pd.DataFrame(point_records.array).astype(dtype_dict)  # , dtype=np.float32)
#frame.drop(columns=list(las.point_format.extra_dimension_names), inplace=True)
#frame_extra = pd.DataFrame(points_extradim.array).astype(np.float32)
#frame = frame.join(frame_extra)
frame_target = pd.DataFrame(points_target.array).astype(np.uint8)
print("DTYPE: ")
print(frame.dtypes)
print(frame_target.dtypes)

with open("./Test/Orne_20130525.csv", 'r') as file:
    #csvfile = csv.reader(file)
    print("row_count : ", sum(1 for line in file)-1)

print("Done")



