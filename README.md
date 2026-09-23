# Text-to-SQL Generator using Fine-Tuned Mistral 7B

A Natural Language to SQL generation project using **Mistral 7B** and **LoRA fine-tuning**.

The project converts natural-language questions about a database into executable SQL queries and compares the performance of:

- Base Mistral 7B
- Fine-tuned Mistral 7B

The generated SQL is then executed against a SQLite database, and the actual database results are compared with the expected results.

---

## 📌 Project Overview

Databases are usually queried using SQL, but not every user knows how to write SQL.

For example, instead of writing:

```sql
SELECT name
FROM employees
WHERE salary > 80000;

a user can simply ask:

What are the names of employees who earn more than 80000?

The system generates the corresponding SQL query automatically.

Basic Pipeline
Natural Language Question
            ↓
       Mistral 7B
            ↓
       Generated SQL
            ↓
      SQLite Database
            ↓
       Query Result
            ↓
   Compare with Expected Result
            ↓
         Accuracy
🎯 Objectives

The main objectives of this project are:

Convert natural-language questions into SQL queries.
Use Mistral 7B as the underlying Large Language Model.
Fine-tune Mistral using LoRA.
Compare the base and fine-tuned models.
Execute generated SQL queries against a real SQLite database.
Evaluate the correctness of generated queries using actual database results.
Build a simple and reproducible Text-to-SQL pipeline.
🧠 Technologies Used
Python
Mistral 7B
Hugging Face Transformers
PEFT
LoRA
PyTorch
BitsAndBytes
SQLite
Pandas
Accelerate
Hugging Face Datasets
🔥 Why LoRA?

Fine-tuning a 7B parameter model directly requires a large amount of GPU memory.

Instead, this project uses LoRA (Low-Rank Adaptation).

LoRA freezes the original model parameters and trains a relatively small number of additional parameters.

This makes fine-tuning much more memory-efficient.

LoRA Configuration
Rank (r): 16
Alpha: 32
Dropout: 0.05
Target Modules:
    q_proj
    k_proj
    v_proj
    o_proj
Bias: none
Task: CAUSAL_LM

Only a small fraction of the original model parameters are trained.

📂 Project Structure
mistral/
│
├── data/
│   ├── __init__.py
│   ├── company.db
│   ├── database.py
│   ├── evaluation_questions.py
│   └── model_outputs.py
│
├── models/
│   └── adapter/
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       ├── tokenizer_config.json
│       └── tokenizer.json
│
├── src/
│   ├── evaluate.py
│   ├── generate_sql.py
│   ├── model.py
│   └── test_database.py
│
├── .gitignore
├── app.py
├── README.md
└── requirements.txt
🗄️ Database

The project uses a small SQLite database for testing Text-to-SQL generation.

The database contains employee and department information.

Example Tables
Employees
employees
--------------------------------
employee_id
name
department_id
salary
experience
Departments
departments
--------------------------------
department_id
department_name

The database allows the model to generate queries involving:

Filtering
Aggregation
Counting
Sorting
Conditions
Joins
Subqueries
💬 Example
Input
What are the names of employees who earn more than 80000?
Generated SQL
SELECT name
FROM employees
WHERE salary > 80000;
Result
Aarav
Diya
Kabir
Sara
⚙️ Model Architecture

The project uses two versions of Mistral for comparison.

1. Base Mistral

The original pretrained Mistral 7B model is used to generate SQL directly from the natural-language question.

Question
   ↓
Base Mistral 7B
   ↓
SQL Query
   ↓
SQLite
2. Fine-Tuned Mistral

A LoRA adapter trained specifically for Text-to-SQL generation is attached to Mistral.

Question
   ↓
Mistral 7B
   +
LoRA Adapter
   ↓
SQL Query
   ↓
SQLite
🔬 Fine-Tuning

The model was fine-tuned using parameter-efficient fine-tuning with LoRA.

Instead of modifying all parameters of Mistral 7B, LoRA trains a small set of additional parameters.

The trained adapter is saved separately.

Important files include:

adapter_config.json
adapter_model.safetensors
tokenizer_config.json
tokenizer.json

The adapter can then be loaded on top of the base Mistral model.

🧪 Evaluation Method

The project does not evaluate the model only by comparing generated SQL strings.

Instead, the generated SQL is actually executed against the SQLite database.

This is important because two different SQL queries can produce the same correct result.

Evaluation Process
Question
   ↓
Generate SQL
   ↓
Execute SQL
   ↓
Get Actual Result
   ↓
Compare with Expected Result
   ↓
Correct / Incorrect

The final accuracy is calculated as:

Accuracy = Correct Queries / Total Queries × 100
📊 Evaluation Results

The current evaluation contains 10 test questions.

Base Mistral
Correct: 10/10
Accuracy: 100%
Fine-Tuned Mistral
Correct: 10/10
Accuracy: 100%
Comparison
Model	Correct	Accuracy
Base Mistral	10/10	100%
Fine-Tuned Mistral	10/10	100%

Both models currently achieve 100% on this small evaluation set.

This does not mean that fine-tuning provides no benefit. The evaluation dataset is small and relatively simple, so a larger and more challenging benchmark would be required to properly measure the improvement from fine-tuning.

🧩 Types of Questions Tested

The evaluation includes questions involving:

Filtering
WHERE salary > 80000
Counting
SELECT COUNT(*)
FROM employees;
Aggregation
SELECT AVG(salary)
FROM employees;
Maximum Value
SELECT MAX(salary)
FROM employees;
Joins
SELECT T1.name
FROM employees AS T1
JOIN departments AS T2
ON T1.department_id = T2.department_id;
Multiple Conditions
WHERE salary > 80000
AND experience > 10;
Subqueries
SELECT AVG(salary)
FROM employees
WHERE department_id = (
    SELECT department_id
    FROM departments
    WHERE department_name = 'Engineering'
);
Sorting and Limiting
ORDER BY salary DESC
LIMIT 1;
🚀 How to Run
1. Clone the Repository
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd mistral
2. Install Dependencies
pip install -r requirements.txt
3. Test the Database
python src/test_database.py
4. Generate SQL
python src/generate_sql.py
5. Run Evaluation
python src/evaluate.py
📁 Important Files
src/model.py

Responsible for loading the base Mistral model and the fine-tuned LoRA adapter.

src/generate_sql.py

Generates SQL queries from natural-language questions.

src/evaluate.py

Evaluates both models by:

Generating SQL
Executing SQL
Comparing results
Calculating accuracy
src/test_database.py

Tests the SQLite database and verifies that queries can be executed correctly.

data/database.py

Contains database-related functionality.

data/evaluation_questions.py

Contains the questions and expected outputs used during evaluation.

data/company.db

SQLite database used for testing.

🛠️ Error Handling

The project includes SQL execution error handling.

If an invalid SQL query is generated, the execution function catches the exception instead of crashing the complete evaluation.

Example:

try:
    cursor.execute(sql)
    return cursor.fetchall()
except Exception as e:
    return f"SQL Error: {e}"
⚠️ Limitations

The current implementation has several limitations:

The evaluation dataset is small.
The database schema is simple.
The test questions are relatively straightforward.
The model may struggle with complex joins and nested queries.
SQL generation quality depends on the training data.
The base and fine-tuned models currently achieve the same accuracy on the small test set.
More diverse databases are required for a stronger evaluation.
🔮 Future Improvements

Possible improvements include:

Evaluate on larger Text-to-SQL datasets such as WikiSQL or Spider.
Add more complex database schemas.
Test nested SQL queries.
Add multiple-table joins.
Add query validation before execution.
Add SQL syntax correction.
Improve the fine-tuning dataset.
Compare different prompting strategies.
Add a web interface for users.
Add support for multiple databases.
Compare LoRA fine-tuning with other parameter-efficient methods.
Add more comprehensive evaluation metrics.
🌐 Future Application

The project can be extended into a user-friendly application where users can connect a database and ask questions using natural language.

For example:

User:
"Show me the average salary in the Engineering department."

        ↓

Text-to-SQL Model

        ↓

SELECT AVG(salary)
FROM employees
WHERE department_id = (...);

        ↓

Database

        ↓

₹95,000

This can make database interaction accessible to users without SQL knowledge.

📚 Concepts Demonstrated

This project demonstrates practical understanding of:

Large Language Models
Mistral 7B
Natural Language Processing
Text-to-SQL
Prompt Engineering
Fine-Tuning
LoRA
Parameter-Efficient Fine-Tuning (PEFT)
Quantization
Hugging Face Transformers
PyTorch
SQLite
SQL
Model Evaluation
Git and GitHub
👩‍💻 Author

Prisha Singh

B.Tech – Artificial Intelligence & Machine Learning