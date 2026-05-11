SELECT
    id AS text_id,
    CONCAT_WS('\n\n', title, description) AS text
FROM issues
WHERE (title IS NOT NULL AND title <> '')
   OR (description IS NOT NULL AND description <> '')
    LIMIT 5000;
