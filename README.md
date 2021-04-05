# Cell Locator json file(s) converter

## Render annotation in json file(s) to image volume
Directory VisualizationDraft contains Python scripts to convert [Cell Locator](https://github.com/BICCN/cell-locator) json file(s) to a ITK compatible volume mask (eg nrrd, nii.gz)

> python convert_json.py --json_name input.json --out_name output.nii.gz --atlas_name average_template_10.nrrd

If parameter json_name is a directory. 

This conversion code is known to work for Cell Locator outputs from April 2020 to Release [0.1.0-2020-07-30](https://github.com/BICCN/cell-locator/releases/tag/0.1.0-2020-07-30)

The json file format changed after the Release 0.1.0-2020-07-30

## Convert new to legacy json format

# Level of Support
We are not currently supporting this code, but simply releasing it to the community AS IS. We are not able to provide any guarantees of support. The community is welcome to submit issues, but you should not expect an active response.

Copyright 2021 Allen Institute
