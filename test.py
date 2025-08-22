import pyodbc

# Simple connection test
connection_strings = [
    "Driver={ODBC Driver 17 for SQL Server};Server=localhost\\SQLEXPRESS06;Database=master;Trusted_Connection=yes;",
    "Driver={ODBC Driver 17 for SQL Server};Server=.\\SQLEXPRESS06;Database=master;Trusted_Connection=yes;",
    "Driver={SQL Server};Server=localhost\\SQLEXPRESS06;Database=master;Trusted_Connection=yes;",
]

print("Testing SQL Server connections...")

for i, conn_str in enumerate(connection_strings, 1):
    try:
        print(f"\nTest {i}: Connecting...")
        conn = pyodbc.connect(conn_str, timeout=10)
        print(f"Test {i}: SUCCESS!")
        print(f"Working connection string: {conn_str}")
        conn.close()
        break
    except Exception as e:
        print(f"Test {i}: FAILED - {str(e)[:100]}")

print("\nTest complete.")