-- staging.transaction_{split}: single column group, enriched with month on train.
CREATE OR REPLACE TABLE `{project}.{staging}.transaction_{split}`
{partition_clause}
CLUSTER BY sample_id
AS
SELECT
  g1.row_id,
  g1.sample_id,
  {month_select}
  g1.seconds_before_predict,
  g1.price,
  g1.volume,
  g1.side
FROM `{project}.{raw}.{split}_transaction_g1` AS g1
{month_join}
