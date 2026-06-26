#standardSQL

-- Kaggle Intro to SQL - Selected Exercises
-- Saved for portfolio/progress tracking.
-- Python setup cells, q.check() cells, and preview-only code are not included.

-- =========================================================
-- Section 1: BigQuery Dataset Exploration
-- Dataset: Chicago Crime
-- =========================================================

-- Task 1: Count how many tables are in the Chicago Crime dataset.
-- My answer: 1 table
-- Table found: crime

-- Task 2: Count how many columns in the crime table have TIMESTAMP data.
-- My answer: 2 TIMESTAMP columns

-- Task 3: Choose fields needed to plot crimes on a map.
-- My answer: latitude, longitude

-- =========================================================
-- Section 2: SELECT / WHERE
-- Dataset: OpenAQ Global Air Quality
-- =========================================================

-- Task 1: Which countries reported pollution levels in units of "ppm"?

SELECT DISTINCT country
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE unit = "ppm";

-- Task 2: Select all rows where pollution value is exactly 0.

SELECT *
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE value = 0;

-- =========================================================
-- Section 3: GROUP BY / HAVING / COUNT
-- Dataset: Hacker News
-- =========================================================

-- Task 1: Find authors with more than 10,000 posts.
-- Output columns: author, NumPosts

SELECT
`by` AS author,
COUNT(1) AS NumPosts
FROM `bigquery-public-data.hacker_news.full`
GROUP BY author
HAVING COUNT(1) > 10000;

-- Task 2: Count deleted comments.
-- My result from the grouped query showed no deleted=True rows, so answer was 0.

SELECT
deleted,
COUNT(1) AS num_posts
FROM `bigquery-public-data.hacker_news.full`
GROUP BY deleted;

-- =========================================================
-- Section 4: ORDER BY and Dates
-- Dataset: World Bank International Education
-- =========================================================

-- Task 1: Find countries with the highest average government education spending
-- as percent of GDP between 2010 and 2017.

SELECT
country_name,
AVG(value) AS avg_ed_spending_pct
FROM `bigquery-public-data.world_bank_intl_education.international_education`
WHERE indicator_code = 'SE.XPD.TOTL.GD.ZS'
AND year >= 2010
AND year <= 2017
GROUP BY country_name
ORDER BY avg_ed_spending_pct DESC;

-- Task 2: Find indicator codes reported by at least 175 rows in 2016.

SELECT
indicator_code,
indicator_name,
COUNT(1) AS num_rows
FROM `bigquery-public-data.world_bank_intl_education.international_education`
WHERE year = 2016
GROUP BY indicator_code, indicator_name
HAVING num_rows >= 175
ORDER BY num_rows DESC;

-- =========================================================
-- Section 5: AS and WITH / CTE
-- Dataset: Chicago Taxi Trips
-- =========================================================

-- Task 1: Find the table name.
-- My answer: taxi_trips

-- Task 2: Peek at the data.
-- Note: Some location fields had NULL / NaN values.

-- Task 3: Count number of taxi trips in each year.

SELECT
EXTRACT(YEAR FROM trip_start_timestamp) AS year,
COUNT(1) AS num_trips
FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
GROUP BY year;

-- Task 4: Count number of taxi trips in each month of 2016.

SELECT
EXTRACT(MONTH FROM trip_start_timestamp) AS month,
COUNT(1) AS num_trips
FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
WHERE EXTRACT(YEAR FROM trip_start_timestamp) = 2016
GROUP BY month;

-- Task 5: For each hour of the day, calculate number of trips and average speed.
-- Date range: after 2016-01-01 and before 2016-04-01.
-- Only valid trips: trip_seconds > 0 and trip_miles > 0.

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
ORDER BY hour_of_day;

-- =========================================================
-- Section 6: JOIN
-- Dataset: StackOverflow
-- =========================================================

-- Task 1: Explore available tables.
-- Important tables used later:
-- posts_questions
-- posts_answers

-- Task 2: Identify relevant columns for joining.
-- posts_questions.tags gives the topic.
-- posts_questions.id is the question ID.
-- posts_answers.parent_id points to the question being answered.
-- posts_answers.owner_user_id gives the answer author's user ID.

-- Task 3: Select StackOverflow questions related to BigQuery.

SELECT
id,
title,
owner_user_id
FROM `bigquery-public-data.stackoverflow.posts_questions`
WHERE tags LIKE '%bigquery%';

-- Task 4: Join questions and answers to get answers to BigQuery-related questions.

SELECT
a.id,
a.body,
a.owner_user_id
FROM `bigquery-public-data.stackoverflow.posts_answers` AS a
INNER JOIN `bigquery-public-data.stackoverflow.posts_questions` AS q
ON a.parent_id = q.id
WHERE q.tags LIKE '%bigquery%';

-- Task 5: Find users who answered BigQuery-related questions and count their answers.

SELECT
a.owner_user_id AS user_id,
COUNT(1) AS number_of_answers
FROM `bigquery-public-data.stackoverflow.posts_answers` AS a
INNER JOIN `bigquery-public-data.stackoverflow.posts_questions` AS q
ON a.parent_id = q.id
WHERE q.tags LIKE '%bigquery%'
GROUP BY user_id
HAVING number_of_answers > 0;

-- Task 6: Generalized expert finder idea.
-- Same logic can be reused by replacing "bigquery" with another topic string.

SELECT
a.owner_user_id AS user_id,
COUNT(1) AS number_of_answers
FROM `bigquery-public-data.stackoverflow.posts_answers` AS a
INNER JOIN `bigquery-public-data.stackoverflow.posts_questions` AS q
ON a.parent_id = q.id
WHERE q.tags LIKE '%python%'
GROUP BY user_id
HAVING number_of_answers > 0
ORDER BY number_of_answers DESC;
