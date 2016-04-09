In each of the 5 data files, there is a header row. Three columns of calendar variables: year, month of the year and day of the month. The last 24 columns are the 24 hours of the day.

In "Load_history.csv", Column A is zone_id ranging from 1 to 20.

In "Temperature_history.csv", Column A is station_id ranging from 1 to 11.

In "submission_template.csv", "weights.csv", and "Benchmark.csv", Column A is id, the identifier for each row; Column B is zone_id ranging from 1 to 21, where the 21st "zone" represents system level, which is the sum of the other 20 zones.

"Benchmark.csv" shows the results from a benchmark model.

Please make sure the submission strictly follow the format as indicated in "submission_template.csv", where the year was sorted in smallest to largest order first, then month, then day, and then zone_id.