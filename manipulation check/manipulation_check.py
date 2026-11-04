import sqlite3
import pandas as pd
import os

# Database path
DB_PATH = "instance/data.db"

# Check whether database exists
if not os.path.exists(DB_PATH):
    print("Database file not found!")
    exit()

# Connect to database
conn = sqlite3.connect(DB_PATH)

print("\n========== DATABASE INSPECTION ==========\n")

# 1. Display all tables
tables = pd.read_sql_query(
    "SELECT name FROM sqlite_master WHERE type='table';",
    conn
)

print("TABLES IN DATABASE:")
print(tables)

# 2. Display columns of each table
for table in tables["name"]:

    print(f"\n========== TABLE: {table} ==========")

    columns = pd.read_sql_query(
        f"PRAGMA table_info('{table}');",
        conn
    )

    print("\nCOLUMNS:")
    print(columns[["name", "type"]])

    # 3. Display number of records
    count = pd.read_sql_query(
        f'SELECT COUNT(*) AS total FROM "{table}";',
        conn
    )

    print("\nTOTAL RECORDS:")
    print(count)

    # 4. Display first five records
    print("\nSAMPLE DATA:")
    sample = pd.read_sql_query(
        f'SELECT * FROM "{table}" LIMIT 5;',
        conn
    )

    print(sample.to_string(index=False))

conn.close()

print("\n========== INSPECTION COMPLETED ==========")
