DELIMITER //
CREATE PROCEDURE CalculateSaldoAll()
BEGIN
    DECLARE done INT DEFAULT FALSE;
    DECLARE current_na2_code INT;
    DECLARE current_date_srok DATE;
    DECLARE current_uploch_n DECIMAL(10, 2);
    DECLARE current_nachislen_n DECIMAL(10, 2);
    DECLARE current_saldo_all DECIMAL(10, 2);

    -- Create a temporary table to store the calculated 'saldo_all' values
    CREATE TEMPORARY TABLE temp_saldo_all (
        na2_code INT,
        date_srok DATE,
        uploch_n DECIMAL(10, 2),
        nachislen_n DECIMAL(10, 2),
        saldo_all DECIMAL(10, 2)
    );

    -- Insert initial data into the temporary table
    INSERT INTO temp_saldo_all (na2_code, date_srok, uploch_n, nachislen_n, saldo_all)
    SELECT na2_code, date_srok, uploch_n, nachislen_n, 0
    FROM nla;

    -- Start recursive calculation
    REPEAT
        SET done = TRUE;
        -- Get the next row from the temporary table
        SELECT na2_code, date_srok, uploch_n, nachislen_n, saldo_all
        INTO current_na2_code, current_date_srok, current_uploch_n, current_nachislen_n, current_saldo_all
        FROM temp_saldo_all
        WHERE saldo_all = 0
        LIMIT 1;

        IF current_na2_code IS NOT NULL THEN
            -- Find the previous date_srok for the current na2_code
            SET current_saldo_all = (
                SELECT IFNULL(SUM(uploch_n - nachislen_n), 0)
                FROM nla
                WHERE na2_code = current_na2_code AND date_srok < current_date_srok
            );

            -- Update the saldo_all in the temporary table
            UPDATE temp_saldo_all
            SET saldo_all = current_saldo_all
            WHERE na2_code = current_na2_code AND date_srok = current_date_srok;

            SET done = FALSE;
        END IF;
    UNTIL done END REPEAT;

    -- Update the 'saldo_all' column in the 'nla' table from the temporary table
    UPDATE nla AS a
    JOIN temp_saldo_all AS b
    ON a.na2_code = b.na2_code AND a.date_srok = b.date_srok
    SET a.saldo_all = b.saldo_all;

    -- Drop the temporary table
    DROP TEMPORARY TABLE IF EXISTS temp_saldo_all;
END;
//
DELIMITER ;

-- Call the stored procedure to calculate 'saldo_all'
CALL CalculateSaldoAll();