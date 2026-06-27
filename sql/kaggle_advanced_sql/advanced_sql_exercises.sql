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

-- =========================================================
-- Section 3: Nested and Repeated Data
-- Dataset: GitHub Repos
-- =========================================================

-- Task 1: Find committers with the most commits in 2016.
-- Output columns: committer_name, num_commits

SELECT
committer.name AS committer_name,
COUNT(1) AS num_commits
FROM `bigquery-public-data.github_repos.sample_commits`
WHERE EXTRACT(YEAR FROM committer.date) = 2016
GROUP BY committer.name
ORDER BY num_commits DESC;

-- Task 2: Determine how many rows are returned after unnesting the sample language table.
-- My answer: 6

SELECT 6 AS num_rows;

-- Task 3: Find programming languages that appear in the most repositories.
-- Output columns: language_name, num_repos

SELECT
l.name AS language_name,
COUNT(1) AS num_repos
FROM `bigquery-public-data.github_repos.languages`,
UNNEST(language) AS l
GROUP BY language_name
ORDER BY num_repos DESC;

-- Task 4: List all languages used in the repository 'polyrabbit/polyglot'.
-- Output columns: name, bytes

SELECT
l.name AS name,
l.bytes AS bytes
FROM `bigquery-public-data.github_repos.languages`,
UNNEST(language) AS l
WHERE repo_name = 'polyrabbit/polyglot'
ORDER BY bytes DESC;

-- =========================================================
-- Section 4: Writing Efficient Queries
-- =========================================================

-- Task 1: Choose which query is most worth optimizing.
-- My answer: 3

SELECT 3 AS query_to_optimize;

-- Task 2: Rewrite the costume location query to filter early and avoid unnecessary large joins.
-- Goal: get the most recent location of each costume owned by a given owner.

WITH LocationsAndOwners AS 
(
SELECT * 
FROM CostumeOwners co INNER JOIN CostumeLocations cl
   ON co.CostumeID = cl.CostumeID
),
LastSeen AS
(
SELECT CostumeID, MAX(Timestamp)
FROM LocationsAndOwners
GROUP BY CostumeID
)
SELECT lo.CostumeID, Location 
FROM LocationsAndOwners lo INNER JOIN LastSeen ls 
    ON lo.Timestamp = ls.Timestamp AND lo.CostumeID = ls.CostumeID
WHERE OwnerID = MitzieOwnerID



-- My answer: 
WITH CurrentOwnersCostumes AS (
SELECT CostumeID
FROM CostumeOwners
WHERE OwnerID = @MitzieOwnerID
),
OwnersCostumesLocations AS (
SELECT
cc.CostumeID,
cl.Timestamp,
cl.Location
FROM CurrentOwnersCostumes AS cc
INNER JOIN CostumeLocations AS cl
ON cc.CostumeID = cl.CostumeID
),
LastSeen AS (
SELECT
CostumeID,
MAX(Timestamp) AS Timestamp
FROM OwnersCostumesLocations
GROUP BY CostumeID
)
SELECT
ocl.CostumeID,
ocl.Location
FROM OwnersCostumesLocations AS ocl
INNER JOIN LastSeen AS ls
ON ocl.CostumeID = ls.CostumeID
AND ocl.Timestamp = ls.Timestamp;
