-- 1. Using only the Invoice table, find how much each customer spent in total.
SELECT  CustomerId, SUM(Total) AS total_spent
FROM Invoice 
Group by CustomerId
Order by total_spent DESC;

-- 2. Using only the Invoice table, find the top 5 customers by total spending.
SELECT CustomerId, SUM(Total) as total_spent
FROM Invoice 
GROUP BY CustomerId
ORDER BY total_spent DESC
LIMIT 5;

-- 3. Using only the Invoice table, count how many invoices each billing country has.
SELECT BillingCountry, COUNT(BillingCountry) as count
FROM Invoice
GROUP BY BillingCountry;

-- 4. Using only the Invoice table, calculate total sales for each billing country.
SELECT BillingCountry, SUM(Total) as total_sales
FROM Invoice
GROUP BY BillingCountry;

-- 5. Using only the Invoice table, find the average invoice total for each billing country.
SELECT BillingCountry, AVG(Total) as avg_total
FROM Invoice
GROUP BY BillingCountry;

-- 6. Using only the Invoice table, show the smallest and largest invoice total for each billing country.
SELECT BillingCountry, MIN(Total) as min_total, MAX(Total) as max_total
FROM Invoice
GROUP BY BillingCountry;

-- 7. Using only the Customer table, count how many customers are from each country.
SELECT Country, COUNT(CustomerId)
FROM Customer
Group by Country;

-- 8. Using only the Invoice table, find customers whose total spending is greater than 40.
SELECT CustomerId, SUM(Total) as total_spent
FROM Invoice
GROUP by CustomerId
HAVING total_spent >40;

-- 9. Using only the Invoice table, find the 5 countries with the highest total invoice sales.
SELECT BillingCountry, SUM(Total) as total_spent
FROM Invoice
GROUP by BillingCountry
Order by total_spent DESC
LIMIT 5;

-- 10. Using only the Invoice table, calculate total sales for each month in 2012.
SELECT 
	strftime('%Y-%m', InvoiceDate) as month, 
	SUM(Total) as total_spent_by_month
FROM Invoice
Where InvoiceDate LIKE '2012%'
GROUP by month
Order by total_spent_by_month

-- 11. Using only the Customer table, count how many customers have a company in each country.
SELECT Country, COUNT(Company) as count
FROM Customer
WHERE Company IS NOT NULL
GROUP BY Country;

-- 12. Using only the Track table, find the 10 genre IDs with the most tracks.
SELECT GenreId, COUNT(TrackId) as per_genre
FROM Track
GROUP BY GenreId
ORDER BY per_genre DESC
LIMIT 10;

-- 13. Using only the Track table, find genre IDs whose average track length is greater than 5 minutes.
SELECT
    GenreId,
    COUNT(TrackId) AS track_count,
    AVG(Milliseconds / 60000.0) AS avg_minutes
FROM Track
GROUP BY GenreId
HAVING AVG(Milliseconds / 60000.0) > 5
ORDER BY avg_minutes DESC;  

-- 14. Using only the InvoiceLine table, find the 10 track IDs with the highest total quantity sold.
SELECT TrackId, SUM(Quantity) sum_quantity
FROM InvoiceLine
GROUP BY TrackId
ORDER BY sum_quantity DESC
LIMIT 10;

-- 15. Using only the InvoiceLine table, group by TrackId and find 10 tracks with highest average unit price 
-- whose average unit price is greater than 1.
SELECT TrackId, AVG(UnitPrice) avg_unit_price
FROM InvoiceLine
GROUP BY TrackId
HAVING AVG(UnitPrice) > 1
ORDER BY avg_unit_price DESC
LIMIT 10;