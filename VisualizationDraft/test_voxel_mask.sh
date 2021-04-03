rm test_20210402_test.nrrd
python convert_json.py --json_name data/ --out_name test_20210402_test.nrrd --atlas_name average_template_10.nrrd
diff test_result/test_20210402.nrrd test_20210402_test.nrrd
