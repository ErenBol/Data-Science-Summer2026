#standardSQL

-- Kaggle Advanced SQL - Selected Exercises
-- Python setup/check cells are not included.

-- =========================================================
-- Section 1: JOINs and UNIONs
-- Dataset: StackOverflow
-- =========================================================

-- Task 1: Include unanswered January 2018 questions.
-- Return question id and time to first answer in seconds.

SELECT
q.id AS q_id,
MIN(TIMESTAMP_DIFF(a.creation_date, q.creation_date, SECOND)) AS time_to_answer
FROM `bigquery-public-data.stackoverflow.posts_questions` AS q
LEFT JOIN `bigquery-public-data.stackoverflow.posts_answers` AS a
ON q.id = a.parent_id
WHERE q.creation_date >= '2018-01-01'
AND q.creation_date < '2018-02-01'
GROUP BY q_id
ORDER BY time_to_answer;

-- Task 2: Initial questions and answers, Part 1.
-- Get each user's first question date and first answer date for January 2019 activity.
-- Include users from both question and answer sides using FULL JOIN.

SELECT 
    q.owner_user_id AS owner_user_id,
    MIN(q.creation_date) AS q_creation_date,
    MIN(a.creation_date) AS a_creation_date
FROM `bigquery-public-data.stackoverflow.posts_questions` AS q
FULL JOIN `bigquery-public-data.stackoverflow.posts_answers` AS a
    ON q.owner_user_id = a.owner_user_id
WHERE q.creation_date >= '2019-01-01'
  AND q.creation_date < '2019-02-01'
  AND a.creation_date >= '2019-01-01'
  AND a.creation_date < '2019-02-01'
GROUP BY owner_user_id;

-- Task 3: For users who joined in January 2019, get their first question date and first answer date.

SELECT
u.id AS id,
MIN(q.creation_date) AS q_creation_date,
MIN(a.creation_date) AS a_creation_date
FROM `bigquery-public-data.stackoverflow.users` AS u
LEFT JOIN `bigquery-public-data.stackoverflow.posts_questions` AS q
ON u.id = q.owner_user_id
LEFT JOIN `bigquery-public-data.stackoverflow.posts_answers` AS a
ON u.id = a.owner_user_id
WHERE u.creation_date >= '2019-01-01'
AND u.creation_date < '2019-02-01'
GROUP BY u.id;

-- Task 4: Get distinct users who posted at least one question or answer on January 1, 2019.

SELECT
q.owner_user_id AS owner_user_id
FROM `bigquery-public-data.stackoverflow.posts_questions` AS q
WHERE DATE(q.creation_date) = '2019-01-01'

UNION DISTINCT

SELECT
p.owner_user_id AS owner_user_id
FROM `bigquery-public-data.stackoverflow.posts_answers` AS p
WHERE DATE(p.creation_date) = '2019-01-01';

-- =========================================================
-- Section 2: Window Functions
-- Dataset: Chicago Taxi Trips
-- =========================================================

-- Task 1: Calculate rolling average of daily taxi trips.
-- Window: current day, previous 3 days, following 3 days.

WITH trips_by_day AS (
SELECT
DATE(trip_start_timestamp) AS trip_date,
COUNT(*) AS num_trips
FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
WHERE trip_start_timestamp > '2016-01-01'
AND trip_start_timestamp < '2016-04-01'
GROUP BY trip_date
ORDER BY trip_date
)
SELECT
trip_date,
AVG(num_trips) OVER (
ORDER BY trip_date
ROWS BETWEEN 3 PRECEDING AND 3 FOLLOWING
) AS avg_num_trips
FROM trips_by_day;

-- Task 2: Number trips within each pickup community area on October 3, 2013.
-- Use RANK().

SELECT
pickup_community_area,
trip_start_timestamp,
trip_end_timestamp,
RANK() OVER (
PARTITION BY pickup_community_area
ORDER BY trip_start_timestamp
) AS trip_number
FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
WHERE DATE(trip_start_timestamp) = '2013-10-03';

-- Task 3: Calculate previous break time in minutes for each taxi trip.
-- Break = current trip start time - previous trip end time for the same taxi.

SELECT
taxi_id,
trip_start_timestamp,
trip_end_timestamp,
TIMESTAMP_DIFF(
trip_start_timestamp,
LAG(trip_end_timestamp) OVER (
PARTITION BY taxi_id
ORDER BY trip_start_timestamp
),
MINUTE
) AS prev_break
FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
WHERE DATE(trip_start_timestamp) = '2013-10-03';
