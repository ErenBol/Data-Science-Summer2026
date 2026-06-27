-- =========================================================
-- LeetCode SQL Practice
-- =========================================================

-- Problem: Second Highest Salary
-- Difficulty: Medium
-- Topic:Advanced String Functions / Regex / Clause
-- Key idea:
-- Select distinct salaries, order them descending, skip the highest one,
-- and return the next one. The outer SELECT returns NULL if no second salary exists.

SELECT
    (
        SELECT DISTINCT salary 
        FROM Employee
        ORDER BY salary DESC
        LIMIT 1 OFFSET 1
    ) AS SecondHighestSalary;


-- Problem:The Number of Employees Which Report to Each Employee
-- Difficulty: Easy
-- Topic:Advanced Select and Joins
-- Key idea:
-- Use a self join: one copy of Employees represents reporting employees,
-- and the other copy represents managers. Match each employee's reports_to
-- value with the manager's employee_id, then group by manager to count direct
-- reports and calculate the average age of those reports.

SELECT 
    m.employee_id,   
    m.name, 
    Count(e.employee_id) as reports_count,
    ROUND(AVG(e.age)) as average_age
FROM Employees e
INNER JOIN Employees m  
ON e.reports_to = m.employee_id
GROUP BY m.employee_id, m.name
ORDER BY m.employee_id;

-- Problem: Managers with at Least 5 Direct Reports
-- Difficulty: Medium
-- Topic: Basic Joins
-- Key idea:
-- Use a self join: one copy of Employee represents managers, and the other
-- copy represents employees reporting to them. Match e.managerId with m.id,
-- group by manager, and keep only managers with at least 5 direct reports.

SELECT
m.name
FROM Employee AS m
INNER JOIN Employee AS e
ON e.managerId = m.id
GROUP BY m.id, m.name
HAVING COUNT(e.id) >= 5
ORDER BY m.name;

-- Problem: Customer Who Visited but Did Not Make Any Transactions
-- Difficulty: Easy
-- Topic: Basic Joins
-- Key idea:
-- Use LEFT JOIN from Visits to Transactions to keep all visits.
-- Visits without matching transactions have NULL transaction fields.
-- Filter those NULL rows and count no-transaction visits per customer.

SELECT
v.customer_id,
COUNT(v.visit_id) AS count_no_trans
FROM Visits AS v
LEFT JOIN Transactions AS t
ON v.visit_id = t.visit_id
WHERE t.visit_id IS NULL
GROUP BY v.customer_id;

-- Problem: Employee Bonus
-- Difficulty: Easy
-- Topic: Basic Joins
-- Key idea:
-- Use LEFT JOIN to keep all employees, including employees without a bonus.
-- Then filter for employees whose bonus is less than 1000 or whose bonus is NULL.

SELECT
e.name,
b.bonus
FROM Employee AS e
LEFT JOIN Bonus AS b
ON e.empId = b.empId
WHERE b.bonus < 1000
OR b.bonus IS NULL;



5. Employee Bonus
6. Rising Temperature
7. Average Time of Process per Machine
8. Rank Scores
