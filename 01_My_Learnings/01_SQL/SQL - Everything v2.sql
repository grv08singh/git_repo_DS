-- DDL: CREATE, ALTER, DROP, TRUNCATE, sp_rename
-- DML: INSERT, UPDATE, DELETE, MERGE
-- DQL: SELECT, JOIN, GROUP BY, HAVING, ORDER BY, DISTINCT, TOP
-- DCL: GRANT, REVOKE, DENY
-- TCL: BEGIN TRANSACTION, COMMIT, ROLLBACK, SAVE TRANSACTION
-- Programmability: Procedures, Functions, Triggers, Views
-- Advanced: CTE, Window Functions, APPLY, Cursors, Dynamic SQL, DMVs
-- Utilities: Temp Tables, Table Variables, CASE, STRING_AGG, STRING_SPLIT




---------------------------------------------------------------
-- Create a database
CREATE DATABASE ECommerceDB;

-- Use a database
USE ecommercedb;

-- Drop database
DROP DATABASE ECommerceDB;
---------------------------------------------------------------

SELECT GETDATE();

---------------------------------------------------------------
-- Create Tables
---------------------------------------------------------------
-- Create Customers Table
DROP TABLE Customers;
CREATE TABLE Customers(
	CustomerID INT IDENTITY(1,1) PRIMARY KEY, --IDENTITY(1,1) is autoincrement 1,2,3,...
	FirstName NVARCHAR(50) NOT NULL,
	LastName NVARCHAR(50) NOT NULL,
	Email NVARCHAR(50) NOT NULL UNIQUE,
	City NVARCHAR(50) NOT NULL DEFAULT 'Unknown',
	JoinDate DATE NOT NULL DEFAULT CAST(GETDATE() AS DATE),
	ContactNumber VARCHAR(15) NULL
);
-- Create Products Table
DROP TABLE Products;
CREATE TABLE Products(
	ProductID INT IDENTITY(1,1) PRIMARY KEY,
	ProductName NVARCHAR(100) NOT NULL,
	Category NVARCHAR(50) NOT NULL,
	Price DECIMAL(10,2) NOT NULL CHECK (Price >= 0),
	StockQty INT NOT NULL CHECK (StockQty >= 0)
);
-- Create Orders Table
DROP TABLE Orders;
CREATE TABLE Orders(
	OrderID INT IDENTITY(1,1) PRIMARY KEY,
	CustomerID INT NOT NULL,
	OrderDate DATETIME NOT NULL DEFAULT GETDATE(),
	TotalAmount DECIMAL(12,2) NOT NULL CHECK (TotalAmount >= 0),
	Status NVARCHAR(20) NOT NULL DEFAULT 'Pending'
);
-- Create OrderItems Table
DROP TABLE OrderItems;
CREATE TABLE OrderItems(
	ItemID INT IDENTITY(1,1) PRIMARY KEY,
	OrderID INT NOT NULL,
	ProductID INT NOT NULL,
	Quantity INT NOT NULL CHECK (Quantity >= 0),
	UnitPrice DECIMAL(10,2) NOT NULL CHECK (UnitPrice >= 0)
)
-- Create Employees Table
DROP TABLE Employees;
CREATE TABLE Employees(
	EmpID INT IDENTITY(1,1) PRIMARY KEY,
	Name NVARCHAR(100) NOT NULL,
	Department NVARCHAR(50) NOT NULL,
	Salary DECIMAL(12,2) NOT NULL,
	ManagerID INT NULL
)
-- Create OrdersArchive Table
DROP TABLE OrdersArchive;
CREATE TABLE OrdersArchive(
	OrderID INT,
	CustomerID INT,
	OrderDate DATETIME,
	TotalAmount DECIMAL(12,2),
	Status NVARCHAR(20)
)
-- Create OrderAutditLog Table
DROP TABLE OrderAuditLog;
CREATE TABLE OrderAuditLog(
	LogID INT IDENTITY(1,1) PRIMARY KEY,
	OrderID INT,
	CustomerID INT,
	Action VARCHAR(20),
	ActionDate DATETIME DEFAULT GETDATE()
)


---------------------------------------------------------------
-- add a column to an existing table
---------------------------------------------------------------
ALTER TABLE Customers
ADD AlternateEmail NVARCHAR(100) NULL;


---------------------------------------------------------------
-- modify a column data type / Change a column data type
---------------------------------------------------------------
ALTER TABLE Customers
ALTER COLUMN AlternateEmail NVARCHAR(50) NULL;


---------------------------------------------------------------
-- rename a column / change column name using stored procedure
---------------------------------------------------------------
EXEC sp_rename
	'Customers.AlternateEmail',
	'BackupEmail',
	'COLUMN';


---------------------------------------------------------------
-- drop a column
---------------------------------------------------------------
ALTER TABLE Customers
DROP COLUMN BackupEmail;


---------------------------------------------------------------
-- difference between DROP, DELETE, and TRUNCATE?
---------------------------------------------------------------
-- DELETE removes rows and can use WHERE.
-- TRUNCATE removes all rows quickly and resets identity.
-- DROP removes the object (TABLE) itself.

-- Example:
-- DELETE FROM dbo.Customers WHERE City = 'Delhi';
-- TRUNCATE TABLE dbo.OrdersArchive;
-- DROP TABLE dbo.OrdersArchive;

---------------------------------------------------------------
-- Add Constraints
---------------------------------------------------------------

-- Drop Foreign Key on Orders(CutomerID) - Customers(CustomerID)
ALTER TABLE Orders 
DROP CONSTRAINT FK_Orders_Customers;
-- Add Foreign Key on Orders(CutomerID) - Customers(CustomerID)
ALTER TABLE Orders 
ADD CONSTRAINT FK_Orders_Customers
FOREIGN KEY (CustomerID)
REFERENCES Customers(CustomerID);

-- Drop FK on OrderItems(OrderID) - Orders(OrderID)
ALTER TABLE OrderItems
DROP CONSTRAINT FK_OrderItems_Orders;
-- Add FK on OrderItems(OrderID) - Orders(OrderID)
ALTER TABLE OrderItems
ADD CONSTRAINT FK_OrderItems_Orders
FOREIGN KEY (OrderID)
REFERENCES Orders(OrderID);

-- Drop FK on OrderItems(ProductID) - Products(ProductID)
ALTER TABLE OrderItems
DROP CONSTRAINT FK_OrderItems_Products;
-- Add FK on OrderItems(ProductID) - Products(ProductID)
ALTER TABLE OrderItems
ADD CONSTRAINT FK_OrderItems_Products
FOREIGN KEY (ProductID)
REFERENCES Products(ProductID);

-- Drop FK on Employees(ManagerID) - Employees(EmpID)
ALTER TABLE Employees
DROP CONSTRAINT FK_Employees_Employees;
-- Add FK on Employees(ManagerID) - Employees(EmpID)
ALTER TABLE Employees
ADD CONSTRAINT FK_Employees_Employees
FOREIGN KEY (ManagerID)
REFERENCES Employees(EmpID);

-- Drop check constraints
ALTER TABLE Products
DROP CONSTRAINT chk_Products_MaxPrice;
-- create check constraints
ALTER TABLE Products
ADD CONSTRAINT chk_Products_MaxPrice CHECK(Price <= 1000000);


-------------------------------------------
-- add a foreign key with cascade behavior
-- changes in parent table ripple down...
-- ...to related rows in child table
-------------------------------------------
CREATE TABLE TempParent (
    ParentID INT PRIMARY KEY
);

CREATE TABLE TempChild (
    ChildID INT PRIMARY KEY,
    ParentID INT,
    CONSTRAINT FK_TempChild_TempParent
    FOREIGN KEY (ParentID) REFERENCES TempParent(ParentID)
    ON DELETE CASCADE
);

DROP TABLE TempChild;
DROP TABLE TempParent;



---------------------------------------------------------------
-- Enter Sample Data into Tables
---------------------------------------------------------------

-- Enter Data into Customers Table
INSERT INTO Customers(FirstName, LastName, Email, City, ContactNumber)
VALUES
('Amit', 'Bansal', 'a_b@gmail.com', 'New Delhi', 123456789),
('Brinda', 'Malhotra', 'b_m@gmail.com', 'Kotkata', 234567891),
('Christopher', 'Nolan', 'c_n@gmail.com', 'Mumbai', 345678912),
('Durga', 'Bai', 'd_b@gmail.com', 'Hyderabad', 456789123),
('Estilla', 'Maria', 'e_m@gmail.com', 'Chennai', 567891234);

-- Enter Data into Products Table
INSERT INTO Products (ProductName, Category, Price, StockQty)
VALUES
('Laptop', 'Electronics', 75000, 50),
('Mouse', 'Electronics', 800, 200),
('Keyboard', 'Electronics', 1500, 120),
('Desk Chair', 'Furniture', 12000, 30),
('Notebook', 'Stationery', 120, 500),
('Pen Pack', 'Stationery', 80, 800);

-- Enter Data into Employees Table
INSERT INTO Employees (Name, Department, Salary, ManagerID)
VALUES
('Raj Malhotra', 'Sales', 90000, NULL),
('Neha Arora', 'Sales', 55000, 1),
('Karan Bhatia', 'IT', 95000, NULL),
('Ishita Jain', 'IT', 65000, 3),
('Rohit Nanda', 'HR', 50000, NULL);

-- Enter Data into Orders Table
INSERT INTO Orders (CustomerID, OrderDate, TotalAmount, Status)
VALUES
(1, '2025-01-15', 75800, 'Completed'),
(2, '2025-02-10', 12000, 'Pending'),
(1, '2025-03-05', 1600, 'Completed'),
(3, '2025-03-20', 240, 'Cancelled'),
(4, '2025-04-01', 75000, 'Completed');

INSERT INTO OrderItems (OrderID, ProductID, Quantity, UnitPrice)
VALUES
(1, 1, 1, 75000),
(1, 2, 1, 800),
(2, 4, 1, 12000),
(3, 3, 1, 1500),
(3, 6, 1, 80),
(4, 5, 2, 120),
(5, 1, 1, 75000);



---------------------------------------------------------------
-- Create View
---------------------------------------------------------------
DROP VIEW vw_CustomerOrders;

CREATE VIEW vw_CustomerOrders
AS
SELECT
	c.CustomerID,
	c.FirstName + ' ' + c.LastName AS FullName,
	o.OrderID,
	o.OrderDate,
	o.TotalAmount,
	o.Status
FROM Customers c
INNER JOIN Orders o
	ON c.CustomerID = o.CustomerID;


---------------------------------------------------------------
-- Query View
---------------------------------------------------------------
SELECT * FROM vw_CustomerOrders;


---------------------------------------------------------------
-- Create Index / Create Indices
---------------------------------------------------------------
DROP INDEX ix_Orders_OrderID ON Orders;

CREATE CLUSTERED INDEX ix_Orders_OrderID	--since OrderID is a PK
ON Orders(OrderID);							--its already a Clustered Ix.


DROP INDEX ix_Cutomers_Email ON Customers;

CREATE NONCLUSTERED INDEX ix_Cutomers_Email
ON Customers(Email);


DROP INDEX ix_Orders_OrderDate ON Orders;

CREATE NONCLUSTERED INDEX ix_Orders_OrderDate
ON Orders(OrderDate DESC);


---------------------------------------------------------------
-- Create Stored Procedures
---------------------------------------------------------------

-----------------------------
-- Get Customer Orders
-----------------------------
-- Drop
DROP PROCEDURE usp_GetCustomerOrders;
-- OR
DROP PROC usp_GetCustomerOrders;

-- Create
CREATE PROCEDURE usp_GetCustomerOrders
	@CustomerID INT,
	@Status NVARCHAR(20) = 'Completed'
AS
BEGIN
	SET NOCOUNT ON;

	SELECT
		OrderId,
		OrderDate,
		TotalAmount,
		Status
	FROM Orders
	WHERE CustomerID = @CustomerID
		AND Status = @Status
	ORDER BY OrderDate DESC;
END;


-----------------------------
-- Procedure: Get Order Count
-----------------------------
DROP PROC usp_GetOrderCount;

CREATE PROC usp_GetOrderCount
	@CustomerID INT,
	@OrderCount INT OUTPUT
AS
BEGIN
	SET NOCOUNT ON;

	SELECT @OrderCount = COUNT(*)
	FROM Orders
	WHERE CustomerID = @CustomerID;
END;


-----------------------------
-- Procedure: Place Order
-----------------------------
DROP PROCEDURE usp_PlaceOrder;

CREATE PROCEDURE usp_PlaceOrder
	@CustomerID INT,
	@ProductID INT,
	@Quantity INT
AS
BEGIN
	SET NOCOUNT ON;

	DECLARE @Price DECIMAL(10,2);
	DECLARE @OrderID INT;

	BEGIN TRY
		BEGIN TRANSACTION;
			-- Get Price from Products Table
			SELECT @Price = Price
			FROM Products
			WHERE ProductID = @ProductID;

			-- insert into Orders table
			INSERT INTO Orders
				(CustomerID, OrderDate, TotalAmount, Status)
			VALUES
				(@CustomerID, GETDATE(), @Price * @Quantity, 'Pending');

			-- insert into OrderItems table
			SET @OrderID = SCOPE_IDENTITY();
			INSERT INTO OrderItems
				(OrderID, ProductID, Quantity, UnitPrice)
			VALUES
				(@OrderID, @ProductID, @Quantity, @Price);

			-- Update Quantity in Products table
			UPDATE Products
			SET StockQty = @Quantity
			WHERE ProductID = @ProductID;
			COMMIT TRANSACTION;
	END TRY
	BEGIN CATCH
		IF @@TRANCOUNT > 0
			ROLLBACK TRANSACTION;
		THROW;
	END CATCH
END;


-----------------------------
-- Procedure: Get User
-----------------------------
DROP PROC usp_GetUser;

CREATE PROC usp_GetUser
	@Name NVARCHAR(50)
AS
BEGIN
	SET NOCOUNT ON;

	SELECT *
	FROM Customers
	WHERE FirstName = @Name;
END;




---------------------------------------------------------------
-- Create User Defined Functions
---------------------------------------------------------------
-----------------------------
-- Get Customer Full Name
-----------------------------
DROP FUNCTION IF EXISTS fn_GetCustomerFullName;

CREATE FUNCTION fn_GetCustomerFullName(
	@CustomerID INT
)
RETURNS NVARCHAR(101)
AS
BEGIN
	DECLARE @FullName NVARCHAR(101);
	
	SELECT @FullName = FirstName + ' ' + LastName
	FROM Customers
	WHERE CustomerID = @CustomerID;

	RETURN @FullName;
END;

-----------------------------
-- Get Orders by City
-----------------------------
DROP FUNCTION IF EXISTS fn_GetOrdersByCity;

CREATE FUNCTION fn_GetOrdersByCity(
	@City NVARCHAR(50)
)
RETURNS TABLE
AS
RETURN (
	SELECT
		o.OrderID,
		c.CustomerID,
		c.FirstName,
		c.Email,
		o.TotalAmount,
		o.Status
	FROM Orders o
	INNER JOIN Customers c
		ON o.CustomerID = c.CustomerID
	WHERE c.City = @City
);

-----------------------------
-- Get Top Products
-----------------------------
DROP FUNCTION IF EXISTS fn_GetTopProducts;

CREATE FUNCTION fn_GetTopProducts(
	@TopN INT
)
RETURNS @Results TABLE(
	ProductID INT,
	ProductName NVARCHAR(100),
	TotalSold INT,
	Revenue DECIMAL(12,2)
)
AS
BEGIN
	INSERT INTO @Results
		SELECT TOP (@TopN)
			p.ProductID,
			p.ProductName,
			SUM(oi.Quantity) AS TotalSold,
			SUM(oi.Quantity * oi.UnitPrice) AS Revenue
		FROM Products p
		INNER JOIN OrderItems oi
			ON p.ProductID = oi.ProductID
		GROUP BY p.ProductID, p.ProductName
		ORDER BY Revenue DESC;

	RETURN
END;



---------------------------------------------------------------
-- Create Triggers
---------------------------------------------------------------
DROP TRIGGER trg_AfterInsertOrder;

CREATE TRIGGER trg_AfterInsertOrder
ON Orders
AFTER INSERT
AS
BEGIN
	SET NOCOUNT ON;

	INSERT INTO OrderAuditLog
		(OrderID, CustomerID, Action)
	SELECT OrderID, CustomerId, 'INSERT'
	FROM inserted;
END;



DROP TRIGGER trg_InsteadOfDeleteOrder;

CREATE TRIGGER trg_InsteadOfDeleteOrder
ON vw_CustomerOrders
INSTEAD OF DELETE
AS
BEGIN
	SET NOCOUNT ON;

	UPDATE Orders
	SET Status = 'Cancelled'
	WHERE OrderID IN (
		SELECT OrderID
		FROM deleted
	)
END;


-------------------------------------------
-- How do procedures differ from functions?
-- Procedures:
-- 1. Can perform INSERT/UPDATE/DELETE
-- 2. Can use transactions
-- 3. returns NONE or MULTIPLE result sets
-- 4. used with EXEC
-- Functions:
-- 1. READ ONLY nature
-- 2. NO TRANSACTIONS
-- 3. returns SINGLE VALUE or TABLE
-- 4. used in SELECT statement
-------------------------------------------


---------------------------------------------------------------
-- DML
---------------------------------------------------------------

-------------------------------------------
-- insert a single row
-------------------------------------------
INSERT INTO Customers 
	(FirstName, LastName, Email, City, ContactNumber)
VALUES
	('Test', 'User', 'testuser@example.com', 'Gurgaon', '9999999910');
	

-------------------------------------------
-- insert multiple rows
-------------------------------------------
INSERT INTO Products
	(ProductName, Category, Price, StockQty)
VALUES
	('USB Cable', 'Electronics', 250, 150),
	('Whiteboard', 'Office', 2200, 25);



	
-------------------------------------------
-- update rows with a condition
-------------------------------------------
UPDATE Products
SET Price = Price * 0.95
WHERE Category = 'Electronics';


-------------------------------------------
-- update using a subquery
-------------------------------------------
UPDATE Orders
SET Status = 'Flagged'
WHERE CustomerID IN
(
    SELECT CustomerID
    FROM Customers
    WHERE City = 'Unknown'
);


-------------------------------------------
-- delete rows with a condition
-------------------------------------------
DELETE FROM Customers
WHERE Email = 'testuser@example.com';


-------------------------------------------
-- capture deleted rows using OUTPUT *****
-------------------------------------------
DELETE FROM OrderItems
OUTPUT DELETED.ItemID, DELETED.OrderID, DELETED.ProductID
WHERE OrderID IN
(
    SELECT OrderID
    FROM Orders
    WHERE Status = 'Cancelled'
);


-------------------------------------------
-- insert data from another table *****
-------------------------------------------
INSERT INTO OrdersArchive 
	(OrderID, CustomerID, OrderDate, TotalAmount, Status)
SELECT 
	OrderID, CustomerID, OrderDate, TotalAmount, Status
FROM Orders
WHERE Status = 'Completed';


-------------------------------------------
-- perform UPSERT using MERGE *****
-------------------------------------------
MERGE Products AS Target
USING
(
    SELECT
        1000 AS ProductID,
        'Gaming Mouse' AS ProductName,
        'Electronics' AS Category,
        1800.00 AS Price,
        60 AS StockQty
) AS Source
ON Target.ProductID = Source.ProductID
WHEN MATCHED THEN
    UPDATE SET
        ProductName = Source.ProductName,
        Category = Source.Category,
        Price = Source.Price,
        StockQty = Source.StockQty
WHEN NOT MATCHED THEN
    INSERT 
		(ProductID, ProductName, Category, Price, StockQty)
    VALUES 
		(Source.ProductID, Source.ProductName, Source.Category, Source.Price, Source.StockQty);


---------------------------------------------------------------
-- DQL
---------------------------------------------------------------

-------------------------------------------
-- INNER JOIN
-------------------------------------------
SELECT
    c.FirstName,
    o.OrderID,
    o.TotalAmount
FROM Customers c
INNER JOIN Orders o
    ON c.CustomerID = o.CustomerID;

-------------------------------------------
-- LEFT JOIN
-------------------------------------------
SELECT
    c.FirstName,
    o.OrderID
FROM Customers c
LEFT JOIN Orders o
    ON c.CustomerID = o.CustomerID;



-------------------------------------------
-- SELF JOIN
-------------------------------------------
SELECT
    e.Name AS Employee,
    m.Name AS Manager
FROM Employees e
LEFT JOIN Employees m
    ON e.ManagerID = m.EmployeeID;




-------------------------------------------
-- GROUP BY and aggregate
-------------------------------------------
SELECT
    c.City,
    COUNT(o.OrderID) AS TotalOrders,
    SUM(o.TotalAmount) AS Revenue,
    AVG(o.TotalAmount) AS AvgOrderValue
FROM Customers c
INNER JOIN Orders o
    ON c.CustomerID = o.CustomerID
GROUP BY c.City;




-------------------------------------------
-- filter aggregated data using HAVING
-------------------------------------------
SELECT
    c.City,
    SUM(o.TotalAmount) AS Revenue
FROM Customers c
INNER JOIN Orders o
    ON c.CustomerID = o.CustomerID
GROUP BY c.City
HAVING SUM(o.TotalAmount) > 1000;




-------------------------------------------
-- ORDER BY
-------------------------------------------
SELECT
    OrderID,
    TotalAmount
FROM Orders
ORDER BY TotalAmount DESC;


-------------------------------------------
-- non-correlated subquery *****
-------------------------------------------
SELECT *
FROM Products
WHERE Price > (SELECT AVG(Price) FROM Products);


-------------------------------------------
-- correlated subquery *****
-------------------------------------------
SELECT
    c.FirstName,
    c.Email
FROM Customers c
WHERE EXISTS
(
    SELECT 1
    FROM Orders o
    WHERE o.CustomerID = c.CustomerID
      AND o.TotalAmount > 50000
);




-------------------------------------------
-- use DISTINCT
-------------------------------------------
SELECT DISTINCT City
FROM Customers;




-------------------------------------------
-- TOP N
-------------------------------------------
SELECT TOP 3 *
FROM Products
ORDER BY Price DESC;




-------------------------------------------
-- CTE (Common Table Expressions)
-------------------------------------------
-- Basic CTE
WITH cte AS
(
    SELECT
        CustomerID,
        SUM(TotalAmount) AS TotalSpent
    FROM Orders
    GROUP BY CustomerID
)
SELECT *
FROM cte
WHERE TotalSpent > 1000;
GO

-- CTE with join
WITH cte AS
(
    SELECT
        CustomerID,
        SUM(TotalAmount) AS Revenue
    FROM Orders
    GROUP BY CustomerID
)
SELECT
    c.FirstName,
    c.LastName,
    cr.Revenue
FROM Customers c
INNER JOIN cte cr
    ON c.CustomerID = cr.CustomerID;
GO

-- Recursive CTE for hierarchy
WITH EmpHierarchy AS
(
    SELECT
        EmployeeID,
        Name,
        ManagerID,
        0 AS LevelNo
    FROM Employees
    WHERE ManagerID IS NULL

    UNION ALL

    SELECT
        e.EmployeeID,
        e.Name,
        e.ManagerID,
        eh.LevelNo + 1
    FROM Employees e
    INNER JOIN EmpHierarchy eh
        ON e.ManagerID = eh.EmployeeID
)
SELECT *
FROM EmpHierarchy;
GO

-- Q4. Multiple CTEs in one query
WITH CityRevenue AS
(
    SELECT
        c.City,
        SUM(o.TotalAmount) AS Revenue
    FROM Customers c
    INNER JOIN Orders o
        ON c.CustomerID = o.CustomerID
    GROUP BY c.City
),
RankedCities AS
(
    SELECT
        City,
        Revenue,
        RANK() OVER (ORDER BY Revenue DESC) AS RevenueRank
    FROM CityRevenue
)
SELECT *
FROM RankedCities;




-------------------------------------------
-- WINDOW FUNCTION
-------------------------------------------
-- WINDOW FUNCTION: ROW_NUMBER
SELECT
    OrderID,
    CustomerID,
    TotalAmount,
    ROW_NUMBER() OVER (ORDER BY TotalAmount DESC) AS RowNum
FROM Orders;

-- WINDOW FUNCTION: RANK
SELECT
    OrderID,
    TotalAmount,
    RANK() OVER (ORDER BY TotalAmount DESC) AS Rnk
FROM Orders;

-- WINDOW FUNCTION: DENSE_RANK
SELECT
    OrderID,
    TotalAmount,
    DENSE_RANK() OVER (ORDER BY TotalAmount DESC) AS DenseRnk
FROM Orders;

-- WINDOW FUNCTION: NTILE *****
SELECT
    OrderID,
    TotalAmount,
    NTILE(4) OVER (ORDER BY TotalAmount DESC) AS Quartile
FROM Orders;

-- WINDOW FUNCTION: PARTITION BY
SELECT
    CustomerID,
    OrderID,
    TotalAmount,
    RANK() OVER (
		PARTITION BY CustomerID 
		ORDER BY TotalAmount DESC
	) AS RankPerCustomer
FROM Orders;

-- WINDOW FUNCTION: Running total *****
SELECT
    OrderID,
    OrderDate,
    TotalAmount,
    SUM(TotalAmount) OVER
    (
        ORDER BY OrderDate
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS RunningTotal
FROM Orders;

-- WINDOW FUNCTION: LAG and LEAD
SELECT
    OrderID,
    OrderDate,
    TotalAmount,
    LAG(TotalAmount, 1, 0) OVER (ORDER BY OrderDate) AS PrevOrderAmount,
    LEAD(TotalAmount, 1, 0) OVER (ORDER BY OrderDate) AS NextOrderAmount
FROM Orders;

-- WINDOW FUNCTION: FIRST_VALUE and LAST_VALUE *****
SELECT
    CustomerID,
    OrderID,
    OrderDate,
    TotalAmount,
    FIRST_VALUE(TotalAmount) OVER
    (
        PARTITION BY CustomerID
        ORDER BY OrderDate
    ) AS FirstOrderAmount,
    LAST_VALUE(TotalAmount) OVER
    (
        PARTITION BY CustomerID
        ORDER BY OrderDate
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS LastOrderAmount
FROM Orders;




-------------------------------------------
-- PIVOT / UNPIVOT *****
-------------------------------------------
SELECT *
FROM
(
    SELECT
        p.Category,
        YEAR(o.OrderDate) AS OrderYear,
        oi.Quantity * oi.UnitPrice AS Revenue
    FROM Orders o
    INNER JOIN OrderItems oi
        ON o.OrderID = oi.OrderID
    INNER JOIN Products p
        ON oi.ProductID = p.ProductID
) AS SourceData
PIVOT
(
    SUM(Revenue)
    FOR Category IN ([Electronics], [Furniture], [Stationery], [Office])
) AS PivotTable;




-------------------------------------------
-- DCL
-------------------------------------------

-- Q1. How do you create a login and user?
-- Run in a real SQL Server instance with proper security permissions.
-- CREATE LOGIN AnalystUser WITH PASSWORD = 'Str0ng@Pass123';
-- USE ECommerceDB;
-- CREATE USER AnalystUser FOR LOGIN AnalystUser;

-- Q2. How do you GRANT SELECT permission?
-- GRANT SELECT ON dbo.Customers TO AnalystUser;

-- Q3. How do you GRANT multiple permissions?
-- GRANT SELECT, INSERT, UPDATE ON dbo.Products TO AnalystUser;

-- Q4. How do you GRANT EXECUTE on a procedure?
-- GRANT EXECUTE ON dbo.usp_GetCustomerOrders TO AnalystUser;

-- Q5. How do you REVOKE permission?
-- REVOKE INSERT ON dbo.Products FROM AnalystUser;

-- Q6. How do you DENY permission?
-- DENY DELETE ON dbo.Orders TO AnalystUser;

-- Q7. How do you create a role and assign permissions?
-- CREATE ROLE SalesRole;
-- GRANT SELECT, INSERT, UPDATE ON dbo.Orders TO SalesRole;
-- EXEC sp_addrolemember 'SalesRole', 'AnalystUser';

-- Q8. How do you grant schema-level permission?
-- GRANT SELECT ON SCHEMA::dbo TO AnalystUser;




-------------------------------------------
-- TCL
-------------------------------------------
-- Q1. Basic transaction with COMMIT / ROLLBACK
BEGIN TRANSACTION;

    UPDATE Products
    SET StockQty = StockQty - 1
    WHERE ProductID = 1;

    IF @@ERROR <> 0
        ROLLBACK TRANSACTION;
    ELSE
        COMMIT TRANSACTION;

-- Q2. Transaction with savepoint
BEGIN TRANSACTION;

    INSERT INTO Orders (CustomerID, OrderDate, TotalAmount, Status)
    VALUES (1, GETDATE(), 5000, 'Pending');

    SAVE TRANSACTION SP1;

    UPDATE Customers
    SET City = 'Mumbai'
    WHERE CustomerID = 1;

    ROLLBACK TRANSACTION SP1;

COMMIT TRANSACTION;

-- Q3. TRY...CATCH with transaction
BEGIN TRY
    BEGIN TRANSACTION;

    UPDATE Products
    SET StockQty = StockQty - 2
    WHERE ProductID = 2;

    UPDATE Orders
    SET Status = 'Shipped'
    WHERE OrderID = 2;

    COMMIT TRANSACTION;
END TRY
BEGIN CATCH
    IF @@TRANCOUNT > 0
        ROLLBACK TRANSACTION;

    SELECT
        ERROR_NUMBER() AS ErrorNumber,
        ERROR_MESSAGE() AS ErrorMessage;
END CATCH;

-- Q4. Isolation level examples
SET TRANSACTION ISOLATION LEVEL READ COMMITTED;

SET TRANSACTION ISOLATION LEVEL READ UNCOMMITTED;

SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;

SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;

-- Q5. Snapshot isolation
-- ALTER DATABASE ECommerceDB SET ALLOW_SNAPSHOT_ISOLATION ON;
-- SET TRANSACTION ISOLATION LEVEL SNAPSHOT;



---------------------------------------------------------------
-- DATA TYPES functions
---------------------------------------------------------------
SELECT ISNULL('abc', 'DefaultValue')
SELECT ISNULL('NULL', 'DefaultValue')
SELECT ISNULL(NULL, 'DefaultValue')

SELECT COALESCE('abc', NULL, 'ThirdValue')
SELECT COALESCE(NULL, NULL, 'ThirdValue')
SELECT COALESCE(NULL, 'abc', 'ThirdValue')

SELECT NULLIF(1, 0)
SELECT NULLIF(0, 0)
SELECT NULLIF(0, 1)

SELECT CAST('2026-05-10' AS DATE)
SELECT CONVERT(VARCHAR(10), GETDATE(), 103)
SELECT CONVERT(VARCHAR(10), GETDATE(), 104)
SELECT CONVERT(VARCHAR(10), GETDATE(), 105)
SELECT CONVERT(VARCHAR(10), GETDATE(), 106)
SELECT CONVERT(VARCHAR(10), GETDATE(), 107)

SELECT TRY_CAST('abc' AS INT)
SELECT TRY_CAST('abc' AS VARCHAR)
SELECT TRY_CAST('123' AS INT)
SELECT TRY_CAST(123 AS VARCHAR)

SELECT TRY_CONVERT(INT, 'abc')
SELECT TRY_CONVERT(VARCHAR, 'abc')
SELECT TRY_CONVERT(INT, '123')
SELECT TRY_CONVERT(VARCHAR, 123)


---------------------------------------------------------------
-- DATE and TIME functions
---------------------------------------------------------------
SELECT GETDATE()                                   -- current date and time
SELECT GETUTCDATE()                                -- current UTC date and time
SELECT SYSDATETIME()                               -- high precision date and time

SELECT DATEADD(DAY, 3, GETDATE())                  -- add 3 days to current date
SELECT DATEDIFF(MONTH,'2025-01-01',GETDATE())      -- date diff in months

SELECT DATEPART(YEAR, GETDATE())                   -- current year
SELECT DATENAME(WEEKDAY, GETDATE())                -- weekday name

SELECT EOMONTH(GETDATE())                          -- current month end date
SELECT FORMAT(GETDATE(), 'dd-MMM-yyyy')            -- change date format



---------------------------------------------------------------
-- STRING functions
---------------------------------------------------------------
SELECT LEN('Gaurav')                    -- #characters
SELECT DATALENGTH('Gaurav')             -- #bytes

SELECT UPPER('Gaurav')                  -- upper case GAURAV
SELECT LOWER('Gaurav')                  -- lower case gaurav

SELECT LTRIM('   Gaurav')               -- trim spaces from left
SELECT RTRIM('Gaurav   ')               -- trim spaces from right
SELECT TRIM('  Gaurav  ')               -- trim start & end spaces

SELECT SUBSTRING('Gaurav', 1, 3)        -- Gau
SELECT LEFT('Gaurav', 2)                -- Ga
SELECT RIGHT('Gaurav', 4)               -- urav

SELECT CHARINDEX('u', 'Gaurav')         -- position of 'u' in 'Gaurav'

SELECT REPLACE('Gaurav', 'au', 'o')     -- Gorav
SELECT REVERSE('Gaurav')                -- varuaG

SELECT REPLICATE('*', 5)                -- *****
SELECT STUFF('SQL2026',4,4,'Server')    -- 


SELECT VALUE FROM STRING_SPLIT('SQL,Python,PowerBI,Azure', ',')

SELECT STRING_AGG(ProductName, ', ') FROM Products



---------------------------------------------------------------
-- MATHEMATICAL functions
---------------------------------------------------------------
SELECT ROUND(15.678, 2)    -- round to 2 decimal place
SELECT CEILING(15.2)       -- 16
SELECT FLOOR(15.9)         -- 15
SELECT ABS(-100)           -- 100
SELECT POWER(2, 10)        -- 2^10 = 1024
SELECT SQRT(144)           -- 12
SELECT RAND()              -- any random number b/w 0 & 1



---------------------------------------------------------------
-- CASE WHEN THEN END
---------------------------------------------------------------
SELECT
    ProductName,
    Price,
    CASE
        WHEN Price >= 50000 THEN 'Premium'
        WHEN Price >= 1000 THEN 'Mid'
        ELSE 'Budget'
    END AS ProductSegment
FROM Products;




---------------------------------------------------------------
-- SET OPERATORS
---------------------------------------------------------------
-- UNION
SELECT City
FROM Customers
UNION
SELECT Department
FROM Employees;


-- UNION ALL
SELECT City
FROM Customers
UNION ALL
SELECT Department
FROM Employees;



---------------------------------------------------------------