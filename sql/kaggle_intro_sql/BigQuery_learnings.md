# BigQuery Learnings — Kaggle Intro to SQL

## Course summary

This course taught me how to use SQL with Google BigQuery through Kaggle notebooks.

Main topics covered:

* BigQuery dataset/table exploration
* Table schemas
* Writing SQL queries inside Python strings
* Running queries with BigQuery Client
* SELECT / WHERE
* GROUP BY / HAVING / COUNT
* ORDER BY
* Date and timestamp functions
* AS aliases
* WITH / CTEs
* JOINs
* LIKE for text matching
* Safe query limits with `maximum_bytes_billed`

---

## BigQuery structure

BigQuery data is organized like this:

```text
Project
└── Dataset
    └── Table
        ├── Columns
        └── Rows
```

Example:

```text
Project: bigquery-public-data
Dataset: hacker_news
Table: full
```

Full BigQuery table path:

```sql
`project.dataset.table`
```

Example:

```sql
FROM `bigquery-public-data.hacker_news.full`
```

Important: BigQuery table paths use **backticks**, not quotes.

```text
`table_path`  → table / column identifier
'text'        → string value
```

---

## BigQuery Python workflow

Kaggle uses Python to connect to BigQuery.

```python
from google.cloud import bigquery

client = bigquery.Client()
```

The `client` object is used to access datasets, tables, and run queries.

---

## Accessing datasets and tables

Get a dataset:

```python
dataset_ref = client.dataset("dataset_name", project="project_name")
dataset = client.get_dataset(dataset_ref)
```

List tables in a dataset:

```python
tables = list(client.list_tables(dataset))

for table in tables:
    print(table.table_id)
```

Get a table:

```python
table_ref = dataset_ref.table("table_name")
table = client.get_table(table_ref)
```

Preview rows:

```python
client.list_rows(table, max_results=5).to_dataframe()
```

---

## Schema

A table schema shows the structure of a table.

It includes:

```text
column name
data type
NULL allowed or not
description
```

Example:

```python
table.schema
```

Useful for checking column names and data types before writing queries.

---

## Running SQL in BigQuery

A SQL query is written as a Python string:

```python
query = """
        SELECT city
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE country = 'US'
        """
```

Run the query:

```python
query_job = client.query(query)
df = query_job.to_dataframe()
```

Meaning:

```text
client.query(query)  → sends query to BigQuery
.to_dataframe()      → gets result as a pandas DataFrame
```

---

## Safe query limits

BigQuery datasets can be very large.

To avoid scanning too much data:

```python
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

query_job = client.query(query, job_config=safe_config)

df = query_job.to_dataframe()
```

If the query would scan more than the limit, BigQuery cancels it.

---

## SELECT / WHERE

Use `SELECT` to choose columns and `WHERE` to filter rows.

Example:

```sql
SELECT city, country, pollutant, value
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE country = 'US'
```

Use `DISTINCT` to get unique values:

```sql
SELECT DISTINCT country
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE unit = "ppm"
```

---

## GROUP BY / COUNT / HAVING

Use `GROUP BY` to group rows, then aggregate each group.

Common aggregate functions:

```sql
COUNT(1)
SUM(column)
AVG(column)
MIN(column)
MAX(column)
```

Example:

```sql
SELECT 
    `by` AS author,
    COUNT(1) AS NumPosts
FROM `bigquery-public-data.hacker_news.full`
GROUP BY author
HAVING COUNT(1) > 10000
```

`WHERE` filters rows before grouping.

`HAVING` filters groups after grouping.

---

## GROUP BY rule

Every selected column must be either:

```text
1. inside GROUP BY
or
2. inside an aggregate function
```

Valid:

```sql
SELECT parent, COUNT(1) AS NumPosts
FROM `bigquery-public-data.hacker_news.full`
GROUP BY parent
```

Invalid:

```sql
SELECT parent, id, COUNT(1)
FROM `bigquery-public-data.hacker_news.full`
GROUP BY parent
```

because `id` is not grouped or aggregated.

---

## Column aliases with AS

Use `AS` to rename output columns.

```sql
SELECT COUNT(1) AS num_rows
FROM `project.dataset.table`
```

Useful because BigQuery may otherwise create unclear column names like `f0_`.

Example:

```sql
SELECT 
    country_name,
    AVG(value) AS avg_ed_spending_pct
FROM `bigquery-public-data.world_bank_intl_education.international_education`
GROUP BY country_name
```

---

## Reserved keyword column names

Some column names are also SQL keywords.

Example: `by`

Because `BY` is used in `GROUP BY` and `ORDER BY`, write it with backticks:

```sql
SELECT `by` AS author
FROM `bigquery-public-data.hacker_news.full`
```

---

## ORDER BY

Use `ORDER BY` to sort results.

Ascending order:

```sql
ORDER BY column_name
```

Descending order:

```sql
ORDER BY column_name DESC
```

Example:

```sql
SELECT 
    country_name,
    AVG(value) AS avg_ed_spending_pct
FROM `bigquery-public-data.world_bank_intl_education.international_education`
WHERE indicator_code = 'SE.XPD.TOTL.GD.ZS'
  AND year >= 2010
  AND year <= 2017
GROUP BY country_name
ORDER BY avg_ed_spending_pct DESC
```

---

## Date and timestamp basics

BigQuery date/time types:

```text
DATE       → only date
DATETIME   → date + time
TIMESTAMP  → exact point in time, often with timezone
```

Example timestamp:

```text
2016-01-01 12:30:00+00:00
```

---

## EXTRACT

Use `EXTRACT()` to get part of a date or timestamp.

```sql
EXTRACT(YEAR FROM timestamp_column)
EXTRACT(MONTH FROM timestamp_column)
EXTRACT(DAY FROM timestamp_column)
EXTRACT(HOUR FROM timestamp_column)
EXTRACT(DAYOFWEEK FROM timestamp_column)
```

Example:

```sql
SELECT 
    EXTRACT(YEAR FROM trip_start_timestamp) AS year,
    COUNT(1) AS num_trips
FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
GROUP BY year
```

---

## DATE()

Use `DATE()` to convert a timestamp/datetime into only the date.

```sql
DATE(timestamp_column)
```

Example:

```sql
SELECT DATE(block_timestamp) AS trans_date
FROM `bigquery-public-data.crypto_bitcoin.transactions`
```

---

## Date range filtering

For timestamp columns, use `TIMESTAMP()`.

Example:

```sql
WHERE trip_start_timestamp > TIMESTAMP('2016-01-01')
  AND trip_start_timestamp < TIMESTAMP('2016-04-01')
```

This includes rows after Jan 1, 2016 and before Apr 1, 2016.

Using `< TIMESTAMP('2016-04-01')` is better than trying to include all of March manually.

---

## Current date/time functions

These do not take a column as input.

```sql
CURRENT_DATE()
CURRENT_DATETIME()
CURRENT_TIMESTAMP()
```

Example:

```sql
SELECT CURRENT_DATE()
```

To compare a timestamp column with today:

```sql
WHERE DATE(timestamp_column) = CURRENT_DATE()
```

---

## Useful date functions

Difference between dates:

```sql
DATE_DIFF(date1, date2, DAY)
DATE_DIFF(date1, date2, MONTH)
DATE_DIFF(date1, date2, YEAR)
```

Add/subtract dates:

```sql
DATE_ADD(date_column, INTERVAL 7 DAY)
DATE_SUB(date_column, INTERVAL 1 MONTH)
```

Group by month/year:

```sql
DATE_TRUNC(date_column, MONTH)
DATE_TRUNC(date_column, YEAR)
```

Format date as text:

```sql
FORMAT_DATE('%Y-%m', date_column)
```

Parse text into date:

```sql
PARSE_DATE('%Y-%m-%d', text_column)
```

---

## WITH / CTE

A CTE is a temporary table inside a query.

Basic structure:

```sql
WITH temp_table AS (
    SELECT column_name
    FROM `project.dataset.table`
)
SELECT *
FROM temp_table
```

The CTE exists only inside that query.

Example:

```sql
WITH time AS (
    SELECT DATE(block_timestamp) AS trans_date
    FROM `bigquery-public-data.crypto_bitcoin.transactions`
)
SELECT 
    COUNT(1) AS transactions,
    trans_date
FROM time
GROUP BY trans_date
ORDER BY trans_date
```

Meaning:

```text
WITH part   → prepare/clean data
SELECT part → analyze prepared data
```

CTEs make long queries easier to read.

---

## CTE with filtering

Example from taxi trips:

```sql
WITH RelevantRides AS (
    SELECT 
        EXTRACT(HOUR FROM trip_start_timestamp) AS hour_of_day,
        trip_miles,
        trip_seconds
    FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
    WHERE trip_start_timestamp > TIMESTAMP('2016-01-01')
      AND trip_start_timestamp < TIMESTAMP('2016-04-01')
      AND trip_seconds > 0
      AND trip_miles > 0
)
SELECT 
    hour_of_day,
    COUNT(1) AS num_trips,
    3600 * SUM(trip_miles) / SUM(trip_seconds) AS avg_mph
FROM RelevantRides
GROUP BY hour_of_day
ORDER BY hour_of_day
```

This query:

```text
1. selects only relevant rides
2. removes invalid trips
3. groups by hour of day
4. counts trips
5. calculates average speed
```

---

## JOIN

Use `JOIN` to combine data from multiple tables.

Basic structure:

```sql
SELECT 
    a.column_name,
    b.column_name
FROM `project.dataset.table1` AS a
INNER JOIN `project.dataset.table2` AS b
    ON a.matching_column = b.matching_column
```

`ON` tells BigQuery how to match rows.

---

## Table aliases in JOINs

Aliases make queries shorter and clearer.

Example:

```sql
FROM `bigquery-public-data.stackoverflow.posts_answers` AS a
INNER JOIN `bigquery-public-data.stackoverflow.posts_questions` AS q
```

Then use:

```sql
a.owner_user_id
q.tags
a.parent_id
q.id
```

Good habit: use table aliases whenever joining tables.

---

## StackOverflow JOIN example

```sql
SELECT 
    a.id,
    a.body,
    a.owner_user_id
FROM `bigquery-public-data.stackoverflow.posts_answers` AS a
INNER JOIN `bigquery-public-data.stackoverflow.posts_questions` AS q
    ON a.parent_id = q.id
WHERE q.tags LIKE '%bigquery%'
```

Meaning:

```text
posts_questions.id        → question ID
posts_answers.parent_id   → question that the answer belongs to
posts_questions.tags      → question topic
posts_answers.owner_user_id → user who wrote the answer
```

---

## LIKE and wildcard %

Use `LIKE` to search text.

`%` means any number of characters.

Example:

```sql
WHERE tags LIKE '%bigquery%'
```

This matches:

```text
bigquery
google-bigquery
bigquery-sql
sql|bigquery|google-cloud-platform
```

---

## Expert finder query pattern

Find users who answered questions about a topic:

```sql
SELECT 
    a.owner_user_id AS user_id,
    COUNT(1) AS number_of_answers
FROM `bigquery-public-data.stackoverflow.posts_answers` AS a
INNER JOIN `bigquery-public-data.stackoverflow.posts_questions` AS q
    ON a.parent_id = q.id
WHERE q.tags LIKE '%bigquery%'
GROUP BY user_id
HAVING number_of_answers > 0
ORDER BY number_of_answers DESC
```

This pattern can be reused for other topics by changing the text inside `LIKE`.

Example:

```sql
WHERE q.tags LIKE '%python%'
```

---

## Main mistakes to avoid

Do not use quotes for BigQuery table paths.

Wrong:

```sql
FROM 'bigquery-public-data.hacker_news.full'
```

Correct:

```sql
FROM `bigquery-public-data.hacker_news.full`
```

Do not forget to group selected columns.

Wrong:

```sql
SELECT country_name, indicator_name, COUNT(1)
FROM `project.dataset.table`
GROUP BY country_name
```

Correct:

```sql
SELECT country_name, indicator_name, COUNT(1)
FROM `project.dataset.table`
GROUP BY country_name, indicator_name
```

Do not confuse string values and column names.

```sql
`by`     -- column name
'by'     -- text/string
```

Do not use `CURRENT_DATE(column)`.

Correct:

```sql
CURRENT_DATE()
DATE(timestamp_column)
```

---

## Most useful patterns from this course

### Count rows by group

```sql
SELECT 
    group_column,
    COUNT(1) AS num_rows
FROM `project.dataset.table`
GROUP BY group_column
ORDER BY num_rows DESC
```

### Filter groups

```sql
SELECT 
    group_column,
    COUNT(1) AS num_rows
FROM `project.dataset.table`
GROUP BY group_column
HAVING COUNT(1) > 100
```

### Date grouping

```sql
SELECT 
    EXTRACT(YEAR FROM timestamp_column) AS year,
    COUNT(1) AS num_rows
FROM `project.dataset.table`
GROUP BY year
ORDER BY year
```

### CTE cleanup

```sql
WITH cleaned AS (
    SELECT 
        column1,
        column2
    FROM `project.dataset.table`
    WHERE column1 IS NOT NULL
)
SELECT *
FROM cleaned
```

### Join + group

```sql
SELECT 
    q.tags,
    COUNT(1) AS num_answers
FROM `project.dataset.answers` AS a
INNER JOIN `project.dataset.questions` AS q
    ON a.parent_id = q.id
GROUP BY q.tags
ORDER BY num_answers DESC
```

---

## What I completed

I completed the Kaggle Intro to SQL course exercises using BigQuery.

Datasets used:

* Chicago Crime
* OpenAQ Global Air Quality
* Hacker News
* World Bank International Education
* Chicago Taxi Trips
* StackOverflow

Main skills practiced:

* Exploring datasets and schemas
* Writing BigQuery SQL queries
* Filtering rows
* Grouping and aggregating
* Sorting results
* Working with dates and timestamps
* Using aliases
* Writing CTEs
* Joining tables
* Searching text with `LIKE`
* Running queries safely with byte limits
