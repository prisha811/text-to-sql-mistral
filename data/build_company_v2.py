"""
build_company_v2.py - builds data/company_v2.db, the richer evaluation database.

Run from the repo root:   python data/build_company_v2.py

Why a second database?
  The original company.db has only 8 employees, so a wrong query can return the right answer by
  luck (your Q10: `salary > 10000` was true for everyone). This one has 4 related tables, 30
  employees, unequal department sizes, unique salaries (no ties in "highest/lowest" questions),
  an empty department (Legal) and two employees with no project assignments, so join and
  subquery mistakes change the result.

Column names follow your original database (department_id, department_name, employee_id, name,
salary, experience) so the schema looks like the style the model was trained on. The first 8
employees are the same people as in company.db. No randomness: it builds the same DB every time.
"""
import sqlite3
from pathlib import Path

OUT = Path(__file__).resolve().parent / "company_v2.db"

SCHEMA = """
CREATE TABLE departments (
    department_id INTEGER PRIMARY KEY,
    department_name TEXT
);
CREATE TABLE employees (
    employee_id INTEGER PRIMARY KEY,
    name TEXT,
    department_id INTEGER,
    salary INTEGER,
    experience INTEGER,
    hire_year INTEGER,
    city TEXT,
    manager_id INTEGER,
    FOREIGN KEY (department_id) REFERENCES departments(department_id)
);
CREATE TABLE projects (
    project_id INTEGER PRIMARY KEY,
    project_name TEXT,
    department_id INTEGER,
    budget INTEGER,
    status TEXT,
    FOREIGN KEY (department_id) REFERENCES departments(department_id)
);
CREATE TABLE assignments (
    employee_id INTEGER,
    project_id INTEGER,
    hours INTEGER,
    PRIMARY KEY (employee_id, project_id)
);
"""

DEPARTMENTS = [(1, "Engineering"), (2, "HR"), (3, "Marketing"), (4, "Finance"),
               (5, "Sales"), (6, "Support"), (7, "Legal")]          # Legal: no employees, no projects

# (employee_id, name, department_id, salary, experience)
EMPLOYEES = [
    (1, "Aarav", 1, 95000, 12), (2, "Diya", 1, 85000, 8), (3, "Rohan", 2, 65000, 6),
    (4, "Ishita", 3, 72000, 5), (5, "Kabir", 4, 88000, 11), (6, "Sara", 1, 105000, 15),
    (7, "Arjun", 2, 60000, 4), (8, "Meera", 3, 78000, 9),            # the original 8
    (9, "Vikram", 1, 98000, 10), (10, "Neha", 1, 73000, 3), (11, "Kunal", 1, 91000, 7),
    (12, "Pooja", 1, 68000, 2), (13, "Rahul", 1, 79000, 5), (14, "Tanvi", 1, 115000, 13),
    (15, "Anil", 2, 71000, 7), (16, "Sneha", 3, 83000, 8), (17, "Varun", 3, 54000, 1),
    (18, "Divya", 3, 66000, 4), (19, "Nikhil", 4, 92000, 9), (20, "Ritu", 4, 76000, 6),
    (21, "Gaurav", 4, 69000, 3), (22, "Amit", 5, 87000, 9), (23, "Priya", 5, 63000, 3),
    (24, "Karan", 5, 74000, 6), (25, "Anjali", 5, 59000, 2), (26, "Manish", 5, 96000, 11),
    (27, "Isha", 5, 52000, 1), (28, "Harsh", 6, 48000, 2), (29, "Swati", 6, 57000, 4),
    (30, "Yash", 6, 61000, 5),
]
DEPT_HEAD = {1: 1, 2: 3, 3: 4, 4: 5, 5: 22, 6: 28}                  # department_id -> employee_id of head
CITIES = ["Bangalore", "Mumbai", "Delhi", "Pune"]

PROJECTS = [  # (project_id, project_name, department_id, budget, status)
    (1, "Apollo", 1, 300000, "active"), (2, "Borealis", 1, 250000, "completed"),
    (3, "Cobalt", 5, 120000, "active"), (4, "Delta", 5, 80000, "completed"),
    (5, "Ember", 4, 90000, "active"), (6, "Falcon", 3, 150000, "on_hold"),
    (7, "Granite", 3, 60000, "completed"), (8, "Horizon", 6, 70000, "active"),
    (9, "Iris", 1, 200000, "on_hold"), (10, "Jade", 2, 40000, "active"),
]


def build(path=OUT):
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.executemany("INSERT INTO departments VALUES (?, ?)", DEPARTMENTS)

    rows = []
    for emp_id, name, dept, salary, exp in EMPLOYEES:
        head = DEPT_HEAD[dept]
        manager = None if emp_id == head else head
        hire_year = 2010 + (emp_id * 7) % 15
        city = CITIES[(emp_id * 3 + emp_id // 5) % 4]
        rows.append((emp_id, name, dept, salary, exp, hire_year, city, manager))
    conn.executemany("INSERT INTO employees VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
    conn.executemany("INSERT INTO projects VALUES (?, ?, ?, ?, ?)", PROJECTS)

    assignments = {}
    for e in range(1, 29):                                   # employees 29 and 30 get no projects
        for k in range(1 + e % 3):                           # 1 to 3 projects each
            project_id = (e * 3 + k * 4) % 10 + 1
            hours = 10 + (e * 13 + k * 7) % 31
            assignments[(e, project_id)] = hours
    conn.executemany("INSERT INTO assignments VALUES (?, ?, ?)",
                     [(e, p, h) for (e, p), h in assignments.items()])
    conn.commit()
    conn.close()


if __name__ == "__main__":
    build()
    conn = sqlite3.connect(OUT)
    for t in ("departments", "employees", "projects", "assignments"):
        print(f"{t:<12} {conn.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]} rows")
    conn.close()
    print(f"Built {OUT}")
