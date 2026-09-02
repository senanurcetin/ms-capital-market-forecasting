-- staging.label: target ve month. sample_id 0..1257636, ay basina ~17.7k sample.
CREATE OR REPLACE TABLE `{project}.{staging}.label`
PARTITION BY RANGE_BUCKET(month, GENERATE_ARRAY(0, 72, 1))
CLUSTER BY sample_id
AS
SELECT CAST(month AS INT64) AS month, sample_id, target
FROM `{project}.{raw}.train_label`
