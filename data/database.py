import sqlite3


def create_database():
    conn = sqlite3.connect("data/company.db")
    cursor = conn.cursor()

    # Departments table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS departments (
            department_id INTEGER PRIMARY KEY,
            department_name TEXT
        )
    """)

    # Employees table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            employee_id INTEGER PRIMARY KEY,
            name TEXT,
            department_id INTEGER,
            salary INTEGER,
            experience INTEGER,
            FOREIGN KEY (department_id)
                REFERENCES departments(department_id)
        )
    """)

    # Insert departments
    departments = [
        (1, "Engineering"),
        (2, "HR"),
        (3, "Marketing"),
        (4, "Finance")
    ]

    cursor.executemany("""
        INSERT OR IGNORE INTO departments
        VALUES (?, ?)
    """, departments)

    # Insert employees
    employees = [
        (1, "Aarav", 1, 95000, 12),
        (2, "Diya", 1, 85000, 8),
        (3, "Rohan", 2, 65000, 6),
        (4, "Ishita", 3, 72000, 5),
        (5, "Kabir", 4, 88000, 11),
        (6, "Sara", 1, 105000, 15),
        (7, "Arjun", 2, 60000, 4),
        (8, "Meera", 3, 78000, 9)
    ]

    cursor.executemany("""
        INSERT OR IGNORE INTO employees
        VALUES (?, ?, ?, ?, ?)
    """, employees)

    conn.commit()
    conn.close()

    print("Database created successfully!")


if __name__ == "__main__":
    create_database()