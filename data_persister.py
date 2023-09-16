import sqlite3
import paramiko

# SSH tunneling configuration
ssh_host = 'your_remote_host'
ssh_port = 22
ssh_username = 'your_username'
ssh_password = 'your_password'
remote_db_path = '/path/to/remote/database.db'

# Local port to forward through the SSH tunnel
local_port = 8888

# Establish an SSH tunnel to the remote server
ssh_client = paramiko.SSHClient()
ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh_client.connect(ssh_host, ssh_port, ssh_username, ssh_password)
ssh_tunnel = ssh_client.get_transport().open_forward_tunnel(
    ('127.0.0.1', local_port),
    (ssh_host, 22)
)

# Connect to the SQLite database through the SSH tunnel
conn = sqlite3.connect(f"ssh://localhost:{local_port}{remote_db_path}")
cursor = conn.cursor()

# Create a table (CRUD: Create)
cursor.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, email TEXT)''')
conn.commit()

# Insert data (CRUD: Create)
cursor.execute("INSERT INTO users (name, email) VALUES (?, ?)", ('John Doe', 'john@example.com'))
conn.commit()

# Read data (CRUD: Read)
cursor.execute("SELECT * FROM users")
rows = cursor.fetchall()
for row in rows:
    print(row)

# Update data (CRUD: Update)
cursor.execute("UPDATE users SET email = ? WHERE name = ?", ('johndoe@example.com', 'John Doe'))
conn.commit()

# Delete data (CRUD: Delete)
cursor.execute("DELETE FROM users WHERE name = ?", ('John Doe',))
conn.commit()

# Close the database connection and SSH tunnel
cursor.close()
conn.close()
ssh_tunnel.close()
ssh_client.close()
