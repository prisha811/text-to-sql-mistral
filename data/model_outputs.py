model_outputs = [
    {
        "question": "What are the names of employees who earn more than 80000?",
        "base_sql": """
            SELECT T1.name
            FROM employees AS T1
            JOIN departments AS T2
            ON T1.department_id = T2.department_id
            WHERE T1.salary > 80000
        """,
        "finetuned_sql": """
            SELECT T1.name
            FROM employees AS T1
            JOIN departments AS T2
            ON T1.department_id = T2.department_id
            WHERE T1.salary > 80000
        """
    },

    {
        "question": "What are the names of employees who have more than 10 years of experience?",
        "base_sql": """
            SELECT T1.name
            FROM employees AS T1
            JOIN departments AS T2
            ON T1.department_id = T2.department_id
            WHERE T1.experience > 10
        """,
        "finetuned_sql": """
            SELECT T1.name
            FROM employees AS T1
            JOIN departments AS T2
            ON T1.department_id = T2.department_id
            WHERE T1.experience > 10
        """
    },

    {
        "question": "What is the average salary of all employees?",
        "base_sql": """
            SELECT AVG(salary)
            FROM employees
        """,
        "finetuned_sql": """
            SELECT AVG(salary)
            FROM employees
        """
    },

    {
        "question": "How many employees are there?",
        "base_sql": """
            SELECT COUNT(*)
            FROM employees
        """,
        "finetuned_sql": """
            SELECT COUNT(*)
            FROM employees
        """
    },

    {
        "question": "What is the highest salary?",
        "base_sql": """
            SELECT MAX(salary)
            FROM employees
        """,
        "finetuned_sql": """
            SELECT MAX(salary)
            FROM employees
        """
    },

    {
        "question": "What are the names of employees in the Engineering department?",
        "base_sql": """
            SELECT T1.name
            FROM employees AS T1
            JOIN departments AS T2
            ON T1.department_id = T2.department_id
            WHERE T2.department_name = 'Engineering'
        """,
        "finetuned_sql": """
            SELECT T1.name
            FROM employees AS T1
            JOIN departments AS T2
            ON T1.department_id = T2.department_id
            WHERE T2.department_name = 'Engineering'
        """
    },

    {
        "question": "What are the names of employees who earn more than 80000 and have more than 10 years of experience?",
        "base_sql": """
            SELECT name
            FROM employees
            WHERE salary > 80000
            AND experience > 10
        """,
        "finetuned_sql": """
            SELECT name
            FROM employees
            WHERE salary > 80000
            AND experience > 10
        """
    },

    {
        "question": "What is the average salary of employees in the Engineering department?",
        "base_sql": """
            SELECT AVG(salary)
            FROM employees
            WHERE department_id = (
                SELECT department_id
                FROM departments
                WHERE department_name = 'Engineering'
            )
        """,
        "finetuned_sql": """
            SELECT AVG(salary)
            FROM employees
            WHERE department_id = (
                SELECT department_id
                FROM departments
                WHERE department_name = 'Engineering'
            )
        """
    },

    {
        "question": "Which employee has the highest salary?",
        "base_sql": """
            SELECT employee_id, name
            FROM employees
            ORDER BY salary DESC
            LIMIT 1
        """,
        "finetuned_sql": """
            SELECT employee_id, name
            FROM employees
            ORDER BY salary DESC
            LIMIT 1
        """
    },

    {
        "question": "How many employees have more than 5 years of experience?",
        "base_sql": """
            SELECT COUNT(*)
            FROM employees
            WHERE experience > 5
        """,
        "finetuned_sql": """
            SELECT COUNT(*)
            FROM employees
            WHERE experience > 5
        """
    }
]