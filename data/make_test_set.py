"""
make_test_set.py - writes data/test_set.json (held-out questions with GOLD SQL) and validates it.

Run from the repo root (after build_company_v2.py):   python data/make_test_set.py

Each item: {"id", "category", "question", "gold_sql"}. evaluate.py runs the gold SQL on
company_v2.db and compares result sets, so the expected answers are never typed by hand.

Rules the questions follow (so scores mean something):
  - the questions name the columns they want ("return the city and the count"), so a query that
    returns extra columns is a real mistake and not an ambiguity;
  - no ties in "highest / lowest / top N" questions, no empty results;
  - they are written by the project author, not copied from sql-create-context. src/check_leakage.py
    still checks them against the 5,500 training rows, because "I wrote them" is not proof.
"""
import json
import sqlite3
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTS = []


def q(category, question, gold_sql):
    TESTS.append({"id": len(TESTS) + 1, "category": category, "question": question, "gold_sql": gold_sql})


# ---------------------------------------------------------------- simple select / filter
q("simple", "List the names of all departments.", "SELECT department_name FROM departments")
q("simple", "What is the name of the department with department_id 3?", "SELECT department_name FROM departments WHERE department_id = 3")
q("simple", "List the names of employees who live in Pune.", "SELECT name FROM employees WHERE city = 'Pune'")
q("simple", "Show the names of employees earning more than 90000.", "SELECT name FROM employees WHERE salary > 90000")
q("simple", "List the names of projects that are on hold.", "SELECT project_name FROM projects WHERE status = 'on_hold'")
q("simple", "Show the names of employees with less than 3 years of experience.", "SELECT name FROM employees WHERE experience < 3")
q("simple", "Which projects have a budget of at least 150000? Return the project names.", "SELECT project_name FROM projects WHERE budget >= 150000")
q("simple", "List the names of employees who have no manager.", "SELECT name FROM employees WHERE manager_id IS NULL")
# ---------------------------------------------------------------- aggregation
q("aggregation", "How many employees are there in total?", "SELECT COUNT(*) FROM employees")
q("aggregation", "What is the average experience of all employees?", "SELECT AVG(experience) FROM employees")
q("aggregation", "What is the lowest salary?", "SELECT MIN(salary) FROM employees")
q("aggregation", "What is the total budget of all projects?", "SELECT SUM(budget) FROM projects")
q("aggregation", "How many projects are currently active?", "SELECT COUNT(*) FROM projects WHERE status = 'active'")
q("aggregation", "What is the highest salary among employees living in Mumbai?", "SELECT MAX(salary) FROM employees WHERE city = 'Mumbai'")
q("aggregation", "How many different cities do employees live in?", "SELECT COUNT(DISTINCT city) FROM employees")
q("aggregation", "What is the total number of hours logged across all assignments?", "SELECT SUM(hours) FROM assignments")
# ---------------------------------------------------------------- order / limit
q("order_limit", "Name the 3 highest-paid employees, highest first.", "SELECT name FROM employees ORDER BY salary DESC LIMIT 3")
q("order_limit", "List the names of the 5 most experienced employees, most experienced first.", "SELECT name FROM employees ORDER BY experience DESC LIMIT 5")
q("order_limit", "What is the name of the project with the largest budget?", "SELECT project_name FROM projects ORDER BY budget DESC LIMIT 1")
q("order_limit", "Return the name and salary of the lowest-paid employee.", "SELECT name, salary FROM employees ORDER BY salary ASC LIMIT 1")
# ---------------------------------------------------------------- group by
q("group_by", "For each city, return the city and the number of employees living there.", "SELECT city, COUNT(*) FROM employees GROUP BY city")
q("group_by", "For each project status, return the status and the number of projects.", "SELECT status, COUNT(*) FROM projects GROUP BY status")
q("group_by", "For each department_id, return the department_id and the average salary of its employees.", "SELECT department_id, AVG(salary) FROM employees GROUP BY department_id")
q("group_by", "For each department_id, return the department_id and the total budget of its projects.", "SELECT department_id, SUM(budget) FROM projects GROUP BY department_id")
q("group_by", "For each project_id, return the project_id and the total hours assigned to it.", "SELECT project_id, SUM(hours) FROM assignments GROUP BY project_id")
# ---------------------------------------------------------------- having
q("having", "Which cities have more than 7 employees? Return the city names.", "SELECT city FROM employees GROUP BY city HAVING COUNT(*) > 7")
q("having", "Return the department_ids whose average salary is above 85000.", "SELECT department_id FROM employees GROUP BY department_id HAVING AVG(salary) > 85000")
q("having", "Which project_ids have more than 5 employees assigned? Return the project_ids.", "SELECT project_id FROM assignments GROUP BY project_id HAVING COUNT(*) > 5")
q("having", "Return the department_ids that have at least 2 projects.", "SELECT department_id FROM projects GROUP BY department_id HAVING COUNT(*) >= 2")
# ---------------------------------------------------------------- join (2 tables)
q("join", "List each employee's name together with the name of their department.", "SELECT e.name, d.department_name FROM employees e JOIN departments d ON e.department_id = d.department_id")
q("join", "List the names of employees who work in the Engineering department.", "SELECT e.name FROM employees e JOIN departments d ON e.department_id = d.department_id WHERE d.department_name = 'Engineering'")
q("join", "Return the names of the projects that belong to the Marketing department.", "SELECT p.project_name FROM projects p JOIN departments d ON p.department_id = d.department_id WHERE d.department_name = 'Marketing'")
q("join", "What is the average salary of employees in the Sales department?", "SELECT AVG(e.salary) FROM employees e JOIN departments d ON e.department_id = d.department_id WHERE d.department_name = 'Sales'")
q("join", "How many employees work in the Finance or HR departments?", "SELECT COUNT(*) FROM employees e JOIN departments d ON e.department_id = d.department_id WHERE d.department_name IN ('Finance', 'HR')")
q("join", "For each department name, return the department name and its number of employees.", "SELECT d.department_name, COUNT(e.employee_id) FROM departments d JOIN employees e ON e.department_id = d.department_id GROUP BY d.department_name")
q("join", "Return each department name with the total salary of its employees.", "SELECT d.department_name, SUM(e.salary) FROM departments d JOIN employees e ON e.department_id = d.department_id GROUP BY d.department_name")
q("join", "List the names of departments that have no employees.", "SELECT d.department_name FROM departments d LEFT JOIN employees e ON e.department_id = d.department_id WHERE e.employee_id IS NULL")
# ---------------------------------------------------------------- multi-table join (3 tables)
q("multi_join", "List the names of employees assigned to the project named Apollo.", "SELECT e.name FROM employees e JOIN assignments a ON a.employee_id = e.employee_id JOIN projects p ON a.project_id = p.project_id WHERE p.project_name = 'Apollo'")
q("multi_join", "For each project name, return the project name and the number of employees assigned to it.", "SELECT p.project_name, COUNT(a.employee_id) FROM projects p JOIN assignments a ON a.project_id = p.project_id GROUP BY p.project_name")
q("multi_join", "What is the total number of hours worked by the employee named Aarav?", "SELECT SUM(a.hours) FROM assignments a JOIN employees e ON a.employee_id = e.employee_id WHERE e.name = 'Aarav'")
q("multi_join", "Return the distinct names of employees who work on active projects.", "SELECT DISTINCT e.name FROM employees e JOIN assignments a ON a.employee_id = e.employee_id JOIN projects p ON a.project_id = p.project_id WHERE p.status = 'active'")
q("multi_join", "For each department name, return the department name and the total hours logged by its employees.", "SELECT d.department_name, SUM(a.hours) FROM departments d JOIN employees e ON e.department_id = d.department_id JOIN assignments a ON a.employee_id = e.employee_id GROUP BY d.department_name")
q("multi_join", "List the distinct names of employees who work on a project that belongs to a different department than their own.", "SELECT DISTINCT e.name FROM employees e JOIN assignments a ON a.employee_id = e.employee_id JOIN projects p ON a.project_id = p.project_id WHERE p.department_id <> e.department_id")
# ---------------------------------------------------------------- subquery
q("subquery", "List the names of employees who earn more than the average salary.", "SELECT name FROM employees WHERE salary > (SELECT AVG(salary) FROM employees)")
q("subquery", "What is the name of the employee with the highest salary?", "SELECT name FROM employees WHERE salary = (SELECT MAX(salary) FROM employees)")
q("subquery", "List the names of employees who are not assigned to any project.", "SELECT name FROM employees WHERE employee_id NOT IN (SELECT employee_id FROM assignments)")
q("subquery", "List the names of departments that have no projects.", "SELECT department_name FROM departments WHERE department_id NOT IN (SELECT department_id FROM projects)")
q("subquery", "List the names of projects whose budget is above the average project budget.", "SELECT project_name FROM projects WHERE budget > (SELECT AVG(budget) FROM projects)")
q("subquery", "List the names of employees whose salary is higher than the average salary of their own department.", "SELECT name FROM employees e WHERE salary > (SELECT AVG(salary) FROM employees WHERE department_id = e.department_id)")
q("subquery", "What is the name of the department that has the most employees?", "SELECT d.department_name FROM departments d JOIN employees e ON e.department_id = d.department_id GROUP BY d.department_id ORDER BY COUNT(*) DESC LIMIT 1")
q("subquery", "List the names of employees who work on more than one project.", "SELECT name FROM employees WHERE employee_id IN (SELECT employee_id FROM assignments GROUP BY employee_id HAVING COUNT(*) > 1)")
q("subquery", "How many employees earn more than the average salary of the Engineering department?", "SELECT COUNT(*) FROM employees WHERE salary > (SELECT AVG(e.salary) FROM employees e JOIN departments d ON e.department_id = d.department_id WHERE d.department_name = 'Engineering')")
q("subquery", "What is the second highest salary?", "SELECT MAX(salary) FROM employees WHERE salary < (SELECT MAX(salary) FROM employees)")
# ---------------------------------------------------------------- self join
q("self_join", "List each employee's name with their manager's name, only for employees who have a manager.", "SELECT e.name, m.name FROM employees e JOIN employees m ON e.manager_id = m.employee_id")
q("self_join", "How many employees report directly to the employee with employee_id 1?", "SELECT COUNT(*) FROM employees WHERE manager_id = 1")
# ---------------------------------------------------------------- misc: ranges, patterns, years
q("misc", "How many employees have between 5 and 10 years of experience, inclusive?", "SELECT COUNT(*) FROM employees WHERE experience BETWEEN 5 AND 10")
q("misc", "List the names of employees whose name starts with the letter A.", "SELECT name FROM employees WHERE name LIKE 'A%'")
q("misc", "What is the average salary of employees with more than 5 years of experience?", "SELECT AVG(salary) FROM employees WHERE experience > 5")
q("misc", "List the names of employees who were hired after 2020.", "SELECT name FROM employees WHERE hire_year > 2020")
q("misc", "How many employees were hired in 2018?", "SELECT COUNT(*) FROM employees WHERE hire_year = 2018")
q("misc", "What is the earliest hire year among all employees?", "SELECT MIN(hire_year) FROM employees")
q("misc", "Return the names of departments whose employees' total salary exceeds 300000.", "SELECT d.department_name FROM departments d JOIN employees e ON e.department_id = d.department_id GROUP BY d.department_name HAVING SUM(e.salary) > 300000")


# ---------------------------------------------------------------- validation
def validate(db_path):
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    total_rows = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                  for t in ("departments", "employees", "projects", "assignments")}
    problems = []
    dup = [k for k, v in Counter(t["question"] for t in TESTS).items() if v > 1]
    problems += [f"duplicate question: {d}" for d in dup]
    for t in TESTS:
        try:
            rows = conn.execute(t["gold_sql"]).fetchall()
        except Exception as exc:
            problems.append(f"#{t['id']} gold SQL fails: {exc}")
            continue
        if not rows or rows == [(None,)]:
            problems.append(f"#{t['id']} EMPTY/NULL result: {t['question']}")
        elif len(rows) == total_rows["employees"] and len(rows[0]) == 1 and t["category"] == "simple":
            problems.append(f"#{t['id']} returns every employee (too easy to match by luck): {t['question']}")
    conn.close()
    return problems


if __name__ == "__main__":
    db = HERE / "company_v2.db"
    if not db.exists():
        raise SystemExit("Run `python data/build_company_v2.py` first.")
    problems = validate(db)
    if problems:
        print("PROBLEMS FOUND:")
        print("\n".join("  " + p for p in problems))
        raise SystemExit(1)
    out = HERE / "test_set.json"
    out.write_text(json.dumps(TESTS, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(TESTS)} questions to {out}")
    for cat, n in Counter(t["category"] for t in TESTS).items():
        print(f"  {cat:<12} {n}")
    print("All gold queries run, none empty.")
