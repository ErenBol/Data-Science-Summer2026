-- 1. Create a calculated employee code and find employees whose first name contains "an".
SELECT  (EmployeeId + 10) * 10 AS newCode, *
FROM Employee
WHERE FirstName LIKE '%an%';

-- 2. Find Brazilian customers who have a company name.
SELECT *
FROM Customer 
Where Company IS NOT NULL AND Country = 'Brazil';   

-- 3. Show customers without a company from Brazil, Germany, or Canada.
SELECT FirstName, LastName, Company, Country
FROM Customer
WHERE Company IS  NULL
AND Country IN (Brazil, Germany, Canada);


-- 4. Show customers whose email looks professional (ends with .com) and who have a company.
SELECT  FirstName || ' ' || LastName as FullName, Email, Company
FROM Customer
WHERE Email LIKE '%.com' AND Company IS NOT NULL;

-- 5. Find tracks longer than 5 minutes and create a minutes column.
SELECT Milliseconds, TrackID, (Milliseconds / 60000) AS Minutes
FROM Track
WHERE Minutes > 5;

-- 6. Find tracks that are expensive compared to normal 0.99 tracks.
SELECT UnitPrice, Name
FROM Track
Where UnitPrice > 0.99;


-- 7. Find invoices from Brazil with medium-sized totals.
SELECT InvoiceID, BillingCountry, Total
FROM Invoice
WHERE BillingCountry = 'Brazil' AND Total BETWEEN 5 AND 15;


-- 8. Find the top-level employee who reports to someone.
SELECT ReportsTo, FirstName || ' ' || LastName AS FullName
FROM Employee 
WHERE ReportsTo IS NOT NULL;


-- 9. Find tracks with large file sizes and show size in MB.
SELECT Bytes, Bytes / 1024 / 1024 as MB
FROM Track
WHERE MB > 100;

-- 10. Use CASE to label invoice totals.
SELECT Total, InvoiceId,
    CASE 
        WHEN Total >= 15 THEN 'High'
        WHEN Total >= 5 THEN 'Medium'
        ELSE 'Low'
    END AS total_category
FROM Invoice;

