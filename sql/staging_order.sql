-- staging.order_{split}: single column group, enriched with month on train.
CREATE OR REPLACE TABLE `{project}.{staging}.order_{split}`
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
  g1.side,
  g1.order_action
FROM `{project}.{raw}.{split}_order_g1` AS g1
{month_join}
