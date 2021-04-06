rm -r test_out
python convert_new_to_old_json.py --json_name data/multiple_2020-09-18.json --out_name test_out
diff -q test_result_new_to_old test_out

rm  test_out.json
python convert_old_to_new_json.py --json_name test_result_new_to_old --out_name test_out.json
diff test_result_new_to_old/converted.json test_out.json