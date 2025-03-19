DELIMITER $

create procedure selectPerson()
begin
select id,name,is_active
from person;
end$

DELIMITER ;

SET @row_number := 0;
SET @prev_na2_code := NULL;
SET @prev_date_srok := NULL;

UPDATE nla AS t
JOIN (
    SELECT
        row_id,
        na2_code,
        date_srok,
        @row_number := IF(@na2_code = na2_code and @date_srok=last_date_srok, @saldo_all - nachislen_n+uploch_n, 1) AS rank,
        @prev_na2_code := na2_code
        @date_srok := date_srok
        @date_srok := last_date_srok
    FROM
        nla
    ORDER BY date_srok, na2_code
) AS ranked
ON t.na2_code = ranked.na2_code
and t.date_srok = ranked.na2_code
and t.row_id = ranked.row_id
SET t.saldo_all = ranked.rank;
