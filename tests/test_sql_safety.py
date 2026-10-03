"""The read-only guard is the last line of defence before SQL reaches Oracle."""
import pytest

from core import sql_safety


@pytest.mark.parametrize("sql", [
    "SELECT * FROM employees",
    "select first_name from employees where salary > 5000",
    "WITH a AS (SELECT 1 AS x FROM dual) SELECT * FROM a",
])
def test_allows_plain_select(sql):
    assert sql_safety.is_safe_select(sql)


@pytest.mark.parametrize("sql", [
    "DELETE FROM employees",
    "UPDATE employees SET salary = 1",
    "INSERT INTO employees VALUES (1)",
    "DROP TABLE employees",
    "TRUNCATE TABLE employees",
    "SELECT 1 FROM dual; DROP TABLE employees",   # stacked statement
    "SELECT * FROM employees -- trailing comment",  # comment injection
    "SELECT * FROM employees /* hidden */",
    "",
])
def test_blocks_anything_that_is_not_a_single_select(sql):
    assert not sql_safety.is_safe_select(sql)


def test_clean_sql_strips_markdown_fence_and_semicolon():
    raw = "```sql\nSELECT * FROM employees;\n```"
    assert sql_safety.clean_sql(raw) == "SELECT * FROM employees"


def test_clean_sql_keeps_clarify_messages():
    assert sql_safety.clean_sql("CLARIFY: which department?").startswith("CLARIFY:")


def test_limit_is_rewritten_to_oracle_11g_rownum():
    out = sql_safety.enforce_oracle11g_syntax("SELECT * FROM employees LIMIT 5").upper()
    assert "LIMIT" not in out
    assert "ROWNUM" in out


SCHEMA = {"EMPLOYEES": {"EMPLOYEE_ID": "NUMBER", "SALARY": "NUMBER"}}


def test_detects_hallucinated_table():
    assert sql_safety.find_invalid_tables("SELECT * FROM fake_table", SCHEMA) == ["FAKE_TABLE"]


def test_real_table_is_accepted():
    assert sql_safety.find_invalid_tables("SELECT * FROM employees", SCHEMA) == []


def test_detects_hallucinated_column():
    assert ("EMPLOYEES", "BOGUS") in sql_safety.find_invalid_columns(
        "SELECT bogus FROM employees", SCHEMA
    )


def test_extracts_requested_count_in_english_and_roman_urdu():
    assert sql_safety._extract_requested_count("top 7 employees") == 7
    assert sql_safety._extract_requested_count("top paanch employees") == 5
