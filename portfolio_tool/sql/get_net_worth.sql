-- For a given date input & ac, get the net worth
WITH fund AS (
    SELECT
      *,
      CASE
        WHEN ticker LIKE '%cash%' THEN weighted_avg_cost
        ELSE price
      END AS final_price,
      DENSE_RANK() OVER(PARTITION BY as_of_date ORDER BY ingestion_timestamp DESC) AS recent_entry
    FROM
      `portfolio_tool.fund_history`
    WHERE
      as_of_date = '2023-01-31'
    QUALIFY
      recent_entry = 1
),

invested_cap AS (
    SELECT
      *,
      DENSE_RANK() OVER(PARTITION BY as_of_date ORDER BY ingestion_timestamp DESC) AS recent_entry
    FROM
      `portfolio_tool.invested_capital_history`
    WHERE
      as_of_date = '2023-01-31'
      AND account_owner = 'horace'
    QUALIFY
      recent_entry = 1
),

allocated_fund AS (
    SELECT
      ic.as_of_date,
      f.fund_name,
      f.bank,
      f.ticker,
      (f.share_count * f.fx_rate * f.final_price * ic.current_allocation_pct) AS current_value
    FROM
      invested_cap AS ic
    INNER JOIN
      fund AS f
    USING
      (as_of_date)
),

other_investments AS (
    SELECT
      *,
      (share_count * fx_rate * price ) AS current_value,
      DENSE_RANK() OVER(PARTITION BY as_of_date ORDER BY ingestion_timestamp DESC) AS recent_entry
    FROM
      `portfolio_tool.other_investments_history`
    WHERE
      as_of_date = '2023-01-31'
      AND account_owner = 'horace'
    QUALIFY
      recent_entry = 1
),

cash AS (
    SELECT
      *,
      DENSE_RANK() OVER(PARTITION BY as_of_date ORDER BY ingestion_timestamp DESC) AS recent_entry
    FROM
      `portfolio_tool.cash_balance_history`
    WHERE
      as_of_date = '2023-01-31'
      AND account_owner = 'horace'
    QUALIFY
      recent_entry = 1
)

SELECT
  'fund' AS source,
  ticker,
  bank,
  SUM(current_value) AS current_value
FROM
  allocated_fund
GROUP BY
  1,2,3
UNION ALL
SELECT
  'other_investments' AS source,
  ticker,
  bank,
  SUM(current_value) AS current_value
FROM
  other_investments
GROUP BY
  1,2,3
UNION ALL
SELECT
  'cash' AS source,
  'cash' AS ticker,
  bank,
  SUM(amount_usd) AS current_value
FROM
  cash
GROUP BY
  1,2,3