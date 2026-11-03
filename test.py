import sqlite3

conn = sqlite3.connect("instance/data.db")

rows = conn.execute("""
    SELECT id, name
    FROM participants
    ORDER BY id
""").fetchall()

for row in rows:
    print(row)

conn.close()