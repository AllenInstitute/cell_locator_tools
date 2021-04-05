rm test_out.nrrd
python convert_new_to_old_json.py --json_name data/multiple_2020-09-18.json --out_name test_20210402_out.nrrd --atlas_name average_template_10.nrrd
diff test_result/test_20210402.nrrd test_20210402_out.nrrd
