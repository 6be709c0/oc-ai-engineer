import pyodbc  
  
server = 'tcp:jopserver.database.windows.net,1433'  
database = 'jopDB'  
username = 'adminDB'  # Make sure this is correct  
password = '@Simplemdp1234'  # Enter your actual password here  
driver = '{ODBC Driver 18 for SQL Server}'  

# Create the connection string  
conn_string = f'''  
DRIVER={driver};  
SERVER={server};  
DATABASE={database};  
UID={username};  
PWD={password};  
Encrypt=yes;  
TrustServerCertificate=no;  
Connection Timeout=30;  
'''  

try:
    # Connect to the database  
    with pyodbc.connect(conn_string) as conn:  
        print("Successfully connected to the database")  
        cursor = conn.cursor()  

        # Execute a sample query  
        cursor.execute("SELECT * FROM information_schema.tables")  
        for row in cursor:  
            print(row)  

except Exception as e:  
    print("Error occurred:", e)