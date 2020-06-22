import json
import pandas as pd
import SimpleITK as sitk
import numpy as np
import csv
import sys
import os
import glob
import random

#
# Example python code to download combine a directory of binary mask volume 
# and combine them into an ITK segmentation volume each with a unique ID
#
input_directory = sys.argv[1]
output_directory = sys.argv[2]

#input_directory = '/allen/aibs/technology/danielsf/marga_annotations/'
#output_directory = './ecker_output/'


#
# create output directory
#
if not os.path.exists( output_directory ) :
    os.makedirs( output_directory )
    
print( "output_directory: %s" % output_directory )

output_columns = {
'IDX': 'uint16', # Zero-based index 
'-R-': 'uint8', # Red color component (0..255)
'-G-': 'uint8', # Green color component (0..255)
'-B-': 'uint8', # Blue color component (0..255)
'-A-': 'float', # Label transparency (0.00 .. 1.00)
'VIS': 'uint8', # Label visibility (0 or 1)
'MSH': 'uint8', # Label mesh visibility (0 or 1)
'LABEL': 'str' #Label description
}

structures = pd.DataFrame( columns = list(output_columns.keys()) )


#
# open input directory and loop through nrrd files
#
print( "input_directory: %s" % input_directory )
glob_output = sorted(glob.iglob( "%s/*.nrrd" % input_directory ))

# initialize
itksnap_index = 1
output_array = None

for fp in glob_output :

    print( fp )
    
    if itksnap_index == 1 :
        input = sitk.ReadImage(fp)
        input_array = sitk.GetArrayFromImage( input )
        output_array = np.zeros_like( input_array )
        output_array.fill(0)
        
    input = sitk.ReadImage(fp)
    input_array = sitk.GetArrayFromImage( input )
    idx = np.where( input_array > 0 )
    output_array[idx] = itksnap_index
    
    row = {}
    row['IDX'] = itksnap_index
    row['-R-'] = random.randint(0,255)
    row['-G-'] = random.randint(0,255)
    row['-B-'] = random.randint(0,255)
    row['-A-'] = 1
    row['VIS'] = 1
    row['MSH'] = 1    
    row['LABEL'] = os.path.splitext( os.path.basename(fp) )[0]
    
    structures = structures.append(row, ignore_index=True)
    
    itksnap_index += 1

    if itksnap_index > 100 :
        break
    
#
# Create output nrrd file
#
input = sitk.ReadImage(fp)
output = sitk.GetImageFromArray( output_array )
output.CopyInformation( input )
output_volume_file = os.path.join( output_directory, 'combined.nrrd' )
sitk.WriteImage( output, output_volume_file, True )
    
#
# Create an ITK-SNAP label description file
#
output_label_file = os.path.join(output_directory,'itksnap_label.txt')
structures.to_csv( output_label_file, 
                        sep=' ', columns=output_columns, header=False, index=False, 
                        float_format='%.0f', quoting=csv.QUOTE_NONNUMERIC )