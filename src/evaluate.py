import sqlite3

DB_PATH = "data/company.db"


def execute_sql(sql):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute(sql)
        result = cursor.fetchall()
        return result
    except Exception as e:
        return f"SQL Error: {e}"
    finally:
        conn.close()


# your existing evaluation code below this

import sys
import os
import sqlite3

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.model_outputs import model_outputs
from data.evaluation_questions import evaluation_questions


for i in range(len(model_outputs)):
    model_outputs[i]["expected_result"] = evaluation_questions[i]["expected_result"]

def evaluate_model(model_name):
    """
    Evaluate a model by comparing its SQL execution
    result with the expected database result.
    """

    correct = 0
    total = len(model_outputs)

    print("\n" + "=" * 60)
    print(f"EVALUATING: {model_name}")
    print("=" * 60)

    for i, item in enumerate(model_outputs, start=1):

        question = item["question"]

        if model_name == "base":
            sql = item["base_sql"]
        else:
            sql = item["finetuned_sql"]

        expected = item["expected_result"]

        result = execute_sql(sql)

        # Normalize lists so ordering does not affect comparison
        if isinstance(result, list):
            result_normalized = sorted(result, key=str)
            expected_normalized = sorted(expected, key=str)
            is_correct = result_normalized == expected_normalized
        else:
            is_correct = False

        if is_correct:
            correct += 1

        print(f"\nQuestion {i}: {question}")
        print("Generated SQL:")
        print(sql.strip())

        print("\nExpected result:")
        print(expected)

        print("Actual result:")
        print(result)

        print("\nStatus:", "✓ Correct" if is_correct else "✗ Incorrect")

    accuracy = (correct / total) * 100

    print("\n" + "-" * 60)
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("-" * 60)

    return accuracy


if __name__ == "__main__":

    base_accuracy = evaluate_model("base")
    finetuned_accuracy = evaluate_model("finetuned")

    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)

    print(f"Base Mistral Accuracy:       {base_accuracy:.2f}%")
    print(f"Fine-tuned Mistral Accuracy: {finetuned_accuracy:.2f}%")