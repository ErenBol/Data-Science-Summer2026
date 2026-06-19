# Data Science & SQL Practice

This repository contains my practice work while learning data science, Python, and SQL.
It includes Python scripts, Jupyter notebooks, and SQL practice queries using the Chinook SQLite database.

## Repository Structure

```text
DATASCIENCE/
├── notebooks/              # Jupyter notebooks
├── python/                 # Python scripts
├── sql/
│   └── chinook/            # SQL practice using Chinook SQLite database
├── data/                   # Local datasets and database files, not pushed
├── requirements.txt        # Python packages
├── .gitignore
└── README.md
```

## SQL Practice

SQL files are used for practicing database querying and data analysis with the Chinook SQLite database.

Planned topics:

* SELECT statements
* WHERE filtering
* comparison operators
* AND / OR conditions
* LIKE
* IN
* BETWEEN
* IS NULL / IS NOT NULL
* calculated columns
* aliases
* CASE expressions
* JOIN operations
* aggregation
* subqueries


## Python / Data Science Practice

Python files and notebooks are used for practicing basic data analysis skills.

Planned topics:

* pandas
* numpy
* matplotlib
* data cleaning
* exploratory data analysis
* working with Excel and CSV files

## Setup

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.\.venv\Scripts\activate
```

Install required packages:

```bash
pip install -r requirements.txt
```

## Notes

The virtual environment and database files are not pushed to GitHub.

Ignored files include:

```text
.venv/
*.db
*.sqlite
*.sqlite3
__pycache__/
.ipynb_checkpoints/
```

SQL files are pushed to the repository, but the actual Chinook SQLite database file is kept locally.
