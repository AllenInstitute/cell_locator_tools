# Cell Locator json file(s) converter

## Render annotations in json file(s) to image volume

Directory *VisualizationDraft* contains Python scripts to convert [Cell Locator](https://github.com/BICCN/cell-locator) json file(s) to a ITK compatible volume mask (eg nrrd, nii.gz)

> python convert_json.py --json_name input.json --out_name output.nii.gz --atlas_name average_template_10.nrrd

If parameter json_name is a directory, the output volume will contain all the annotations of all json files in the directory each specified with a unique value. Scripts assumes non-overlapping annotations. Areas of overlapped are assigned to only one annotation with no guaranteed ordering.

This conversion code is known to work for annotations on the Allen Mouse CCF and Cell Locator outputs from version [0.1.0-2020-07-30](https://github.com/BICCN/cell-locator/releases/tag/0.1.0-2020-07-30).

The json file format changed after the Release [0.1.0-2020-07-30](https://github.com/BICCN/cell-locator/releases/tag/0.1.0-2020-07-30).

For newer format files, you can convert to legacy format before using rendering script.

## Convert new to legacy json format

Directory *ConvertBetweenJsonVersions* contains Python scripts to convert newer format Cell Locator json files to legacy format.

> python convert_new_to_old_json.py --json_name input.json --out_name output_directory

Each annotation within the input.json is converted to individual json files in the output_directory. The output directory can be used as input for rendering script.

This conversion code is known to work for annotations on the Allen Mouse CCF and Cell Locator outputs from version [0.1.0-2020-09-18](https://github.com/BICCN/cell-locator/releases/tag/0.1.0-2020-09-18).


# Level of Support
We are not currently supporting this code, but simply releasing it to the community AS IS. We are not able to provide any guarantees of support. The community is welcome to submit issues, but you should not expect an active response.

Copyright 2021 Allen Institute
