WITH RECURSIVE cte AS (
  SELECT date_srok, na2_code, nachislen_n, umenshen_n, saldo_all AS max_date, nachislen_n-umenshen_n AS D
  FROM nla
  WHERE date_srok = (SELECT MIN(date_srok) FROM nla)

  UNION ALL

  SELECT t.date_srok, t.na2_code, t.nachislen_n, t.umenshen_n, GREATEST(t.date_srok, cte.max_date), t.nachislen_n + t.umenshen_n + cte.D AS D
  FROM nla t
  JOIN cte ON t.date_srok = cte.date_srok + INTERVAL 1 DAY and t.na2_code=cte.na2_code
)
SELECT * FROM cte;
