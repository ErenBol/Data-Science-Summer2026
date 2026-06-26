# SQL Practice

This folder contains my SQL learning work.

The goal is to practice SQL from basic querying to more advanced analysis using SQLite and BigQuery.

## Folder Structure

```text
sql/
├── README.md
├── chinook/
│   ├── 01_select_where_operators.sql
│   ├── 02_group_by_aggregates.sql
│   └── 03_joins_inner_left.sql
├── kaggle_intro_sql/
│   ├── intro_sql_notes.md
│   ├── intro_sql_exercises.sql
│   └── bigquery_learnings.md
└── kaggle_advanced_sql/
    ├── advanced_sql_notes.md
    ├── advanced_sql_exercises.sql
    └── images/
```

## 1. Chinook SQL Practice

The `chinook/` folder contains SQLite practice queries using the Chinook database.

### Files

```text
01_select_where_operators.sql
```

Focus:

* SELECT
* WHERE
* comparison operators
* AND / OR
* LIKE
* IN
* BETWEEN
* aliases
* calculated columns

```text
02_group_by_aggregates.sql
```

Focus:

* COUNT
* SUM
* AVG
* MIN / MAX
* GROUP BY
* HAVING
* sorting aggregate results

```text
03_joins_inner_left.sql
```

Focus:

* INNER JOIN
* LEFT JOIN
* joining related tables
* understanding unmatched rows
* using table aliases

## 2. Kaggle Intro to SQL

The `kaggle_intro_sql/` folder contains notes and selected exercise solutions from Kaggle Learn's Intro to SQL course.

### Topics learned

* BigQuery projects, datasets, and tables
* Table schemas
* SELECT / WHERE
* GROUP BY / HAVING
* ORDER BY
* date and timestamp functions
* AS aliases
* WITH / CTEs
* INNER JOIN
* LIKE
* safe query limits

### Files

```text
intro_sql_notes.md
```

Short lesson notes.

```text
intro_sql_exercises.sql
```

Selected final exercise queries.

```text
bigquery_learnings.md
```

Summary of BigQuery concepts learned in the course.

## 3. Kaggle Advanced SQL

The `kaggle_advanced_sql/` folder contains notes and selected exercise solutions from Kaggle Learn's Advanced SQL course.

### Topics learned / being practiced

* INNER JOIN
* LEFT JOIN
* RIGHT JOIN
* FULL JOIN
* UNION ALL
* UNION DISTINCT
* analytic/window functions
* PARTITION BY
* ORDER BY inside OVER()
* window frames
* RANK
* LAG
* rolling averages

### Files

```text
advanced_sql_notes.md
```

Short notes from the lessons.

```text
advanced_sql_exercises.sql
```

Selected final exercise queries.

```text
images/
```

Images used in notes, such as JOIN diagrams and window function diagrams.

## Key Comparisons

### JOIN vs UNION

JOIN combines tables horizontally.

UNION combines query results vertically.

```text
JOIN  → add columns
UNION → add rows
```

### INNER JOIN vs LEFT JOIN

INNER JOIN keeps only matching rows.

LEFT JOIN keeps all rows from the left table and fills missing right-side values with NULL.

### GROUP BY vs PARTITION BY

GROUP BY collapses rows into one row per group.

PARTITION BY keeps rows and calculates values within each group.

### ORDER BY inside OVER vs final ORDER BY

ORDER BY inside OVER controls calculation order for the window function.

Final ORDER BY controls the returned result order.

## Current SQL Goal

Build enough SQL skill to:

* explore datasets
* answer analysis questions with queries
* join multiple tables correctly
* summarize data with aggregates
* use CTEs for cleaner logic
* use window functions for row-level analytics
