-- staging.market_{split}: rejoins the column groups on row_id.
-- row_id is the row position in the source feather file. Different column
-- projections of the same file return the same row order, so the positional join
-- is safe (staging.py:assert_group_alignment proves it on the real data).
CREATE OR REPLACE TABLE `{project}.{staging}.market_{split}`
{partition_clause}
CLUSTER BY sample_id
AS
SELECT
  g1.row_id,
  g1.sample_id,
  {month_select}
  g1.seconds_before_predict,
  g1.transaction_avgprice,
  g1.transaction_volume,
  g1.transaction_count,
  g2.ask_price_1, g2.ask_volume_1, g2.bid_price_1, g2.bid_volume_1,
  g3.ask_price_2, g3.ask_volume_2, g3.bid_price_2, g3.bid_volume_2
FROM `{project}.{raw}.{split}_market_g1` AS g1
JOIN `{project}.{raw}.{split}_market_g2` AS g2 USING (row_id)
JOIN `{project}.{raw}.{split}_market_g3` AS g3 USING (row_id)
{month_join}
