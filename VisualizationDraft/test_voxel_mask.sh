rm test_20201007b.nrrd
python convert_json.py --json_name data/ --out_name test_20201007b.nrrd
diff test_result/test_20201007.nrrd test_20201007b.nrrd
