
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


---------------------------------------------------------------
-- DML
---------------------------------------------------------------

-------------------------------------------
-- add a single row
-------------------------------------------
INSERT INTO Customers 
	(FirstName, LastName, Email, City, ContactNumber)
VALUES
	('Test', 'User', 'testuser@example.com', 'Gurgaon', '9999999910');






---------------------------------------------------------------
-- 
---------------------------------------------------------------


---------------------------------------------------------------
-- 
---------------------------------------------------------------


---------------------------------------------------------------
-- 
---------------------------------------------------------------


---------------------------------------------------------------
-- 
---------------------------------------------------------------


---------------------------------------------------------------
-- 
---------------------------------------------------------------


---------------------------------------------------------------
-- 
---------------------------------------------------------------


---------------------------------------------------------------
-- 
---------------------------------------------------------------


---------------------------------------------------------------
-- 
---------------------------------------------------------------


---------------------------------------------------------------
-- 
---------------------------------------------------------------


---------------------------------------------------------------
-- 
---------------------------------------------------------------


---------------------------------------------------------------
-- 
---------------------------------------------------------------


---------------------------------------------------------------
-- 
---------------------------------------------------------------


---------------------------------------------------------------
-- 
---------------------------------------------------------------


---------------------------------------------------------------
-- 
---------------------------------------------------------------


---------------------------------------------------------------
-- 
---------------------------------------------------------------


---------------------------------------------------------------
-- 
---------------------------------------------------------------


---------------------------------------------------------------
-- 
---------------------------------------------------------------


---------------------------------------------------------------
-- 
---------------------------------------------------------------


---------------------------------------------------------------
-- 
---------------------------------------------------------------


---------------------------------------------------------------
-- 
---------------------------------------------------------------


---------------------------------------------------------------
-- 
---------------------------------------------------------------


---------------------------------------------------------------
-- 
---------------------------------------------------------------


---------------------------------------------------------------
-- 
---------------------------------------------------------------


---------------------------------------------------------------
-- 
---------------------------------------------------------------