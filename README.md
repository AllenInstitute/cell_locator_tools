# CellLocator_ROI_Prototype

Directory VisualizationDraft contains Python scripts to convert [Cell Locator](https://github.com/BICCN/cell-locator) json file(s) to a ITK compatible (nrrd, nii.gz) volume mask

> python convert_json.py --json_name input.json --out_name output.nii.gz --atlas_name average_template_10.nrrd


This conversion code is known to work for Cell Locator outputs from April 2020 to Release [0.1.0-2020-07-30](https://github.com/BICCN/cell-locator/releases/tag/0.1.0-2020-07-30)

The json file format changed after the Release 0.1.0-2020-07-30
