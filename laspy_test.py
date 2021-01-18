#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, psutil
import time
import numpy as np
import pandas as pd
import pylas

las = pylas.read("./../../../PostDoc_Temp/AUP1_Orne/130525_1M_ft_trgt.las")
dimensions = list(las.point_format.dimension_names)
extra_dims = list(las.point_format.extra_dimension_names)
point_format = dict()

# Point formats for LAS 1.2 to 1.4
gps_time = ['gps_time']
nir = ['nir']
rgb = ['red', 'green', 'blue']
wavepacket = ['wavepacket_index', 'wavepacket_offset', 'wavepacket_size',
              'return_point_wave_location', 'x_t', 'y_t', 'z_t']

point_format[0] = ['X', 'Y', 'Z', 'intensity', 'return_number', 'number_of_returns',
                   'scan_direction_flag', 'edge_of_flight_line', 'classification',
                   'synthetic', 'key_point', 'withheld', 'scan_angle_rank',
                   'user_data', 'point_source_id']
point_format[1] = point_format[0] + gps_time
point_format[2] = point_format[0] + rgb
point_format[3] = point_format[0] + gps_time + rgb

# Point formats for LAS 1.3 to 1.4
point_format[4] = point_format[0] + gps_time + wavepacket
point_format[5] = point_format[0] + gps_time + rgb + wavepacket

# Point formats for LAS 1.4
point_format[6] = ['X', 'Y', 'Z', 'intensity', 'return_number', 'number_of_returns',
                   'synthetic', 'key_point', 'withheld', 'overlap', 'scanner_channel',
                   'scan_direction_flag', 'edge_of_flight_line', 'classification',
                   'user_data', 'scan_angle_rank', 'point_source_id', 'gps_time']
point_format[7] = point_format[6] + rgb
point_format[8] = point_format[6] + rgb + nir
point_format[9] = point_format[6] + wavepacket
point_format[10] = point_format[6] + rgb + nir + wavepacket

pt_nbr = las.header.point_count
print("Version: {}".format(las.header.version))
print("Point format: {}".format(las.point_format.id))
print("Point count: {}".format(pt_nbr))
print(extra_dims)

frame = pd.DataFrame()
for dim in extra_dims:
    frame[dim] = las[dim]
las = None
print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)

predictions = pd.DataFrame()
predictions['Prediction'] = frame['Target']

print(predictions.shape)
predictions['BestProba'] = frame['CalibIntensity_(1)']
predictions['Proba0'] = frame['CalibIntensity_(2)']
predictions['Proba1'] = frame['CalibIntensity_(5)']
print(predictions.columns.values.tolist())
frame = None

# copy_data = pd.read_csv("./../../../PostDoc_Temp/AUP1_Orne/130525_1M_ft_trgt.csv",
#                         sep=',', header='infer')
#
# final_data = copy_data.join(predictions)
# final_data.to_csv('/test/final_data.csv', sep=',', header=True, index=False)

# test_copy = pylas.read("./../../../PostDoc_Temp/AUP1_Orne/130525_1M_ft_trgt.las")
#
# print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
# test_copy.add_extra_dim(name="Prediction", type='uint8',
#                         description="Prediction done by the model")
# test_copy.add_extra_dim(name="BestProba", type='f4',
#                         description="Best probability")
# test_copy['Prediction'] = predictions['Prediction']
# test_copy['BestProba'] = predictions['BestProba']
#
# for i in ['Proba' + str(cla) for cla in range(0, 2)]:
#     test_copy.add_extra_dim(name=i, type='f4',
#                             description="Probability for this class")
#     test_copy[i] = predictions[i]
#
# test_copy.write('/test/test_copy.las')
# test_copy = None
# print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
