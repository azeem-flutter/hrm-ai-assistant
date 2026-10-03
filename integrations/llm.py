"""
llm.py
------
Builds the (large, very deliberate) system prompt that turns the local
Ollama model into a disciplined Oracle 11g SQL writer, and wraps the
actual call to the Ollama server.
"""

import json
from datetime import date
from typing import Optional

try:
    import ollama
except ImportError:
    # Only a hard requirement when LLM_PROVIDER="ollama" (the default) is
    # actually used -- see call_local_model() below. A Groq-only setup
    # (LLM_PROVIDER="groq") never touches this and doesn't need the
    # ollama package installed at all.
    ollama = None

from config import config


def build_system_prompt(
    schema_text: str,
    all_table_names: list,
    examples: Optional[list] = None,
    other_table_names: Optional[list] = None,
) -> str:
    # `schema_text` is now only the RAG-retrieved subset of tables most
    # relevant to this question — NOT the full schema. But the "row count
    # per table" UNION ALL instruction below must always enumerate every
    # table in the database regardless of what got retrieved, so that
    # feature takes its own explicit `all_table_names` list instead of
    # parsing it out of schema_text like before.
    union_all_example = ""
    if all_table_names:
        union_parts = [
            f"SELECT '{table}' AS table_name, COUNT(*) AS row_count FROM {table}"
            for table in all_table_names
        ]
        union_all_example = "\n  UNION ALL\n  ".join(union_parts)
        union_all_example = f"\n  {union_all_example}"

    examples_block = ""
    if examples:
        example_entries = [
            f'Q: "{ex["question"]}"\nA: {ex["sql"]}' for ex in examples
        ]
        examples_block = (
            "\n*** RELEVANT PAST EXAMPLES — real, correct answers to "
            "similar questions asked before. Follow this exact style and "
            "SQL pattern when the current question matches one of these ***\n\n"
            + "\n\n".join(example_entries)
            + "\n"
        )

    # RAG only sends the top-K tables it judged most relevant to the
    # question's WORDING — if the actually-needed table doesn't happen to
    # match well semantically, it never makes it into `schema_text` above,
    # and the model has no way to know it exists at all. Left with only
    # the retrieved subset, a model under instruction to "use the real
    # schema" will sometimes invent a plausible-sounding table/column
    # name instead of admitting it doesn't have what it needs (this is
    # what produced the RATIO_PACKS.CATEGORY hallucination in testing).
    # This block gives the model visibility into every OTHER real table
    # name (no columns — that would defeat the point of RAG trimming the
    # prompt) so it can recognize "the table I need exists, I just wasn't
    # given its columns" and ask instead of guessing.
    other_tables_block = ""
    if other_table_names:
        other_tables_block = (
            "\n*** OTHER TABLES IN THIS DATABASE — names only, not shown "
            "in detail above because they did not look relevant to this "
            "question ***\n"
            + ", ".join(other_table_names)
            + "\nIf, after reading the question again, you believe it "
              "actually needs one of these tables, do NOT guess its "
              "columns or invent structure for it. Reply with exactly:\n"
              "CLARIFY: <a short question naming that table and asking "
              "what information from it is needed>\n"
              "Never write SQL against a table from this list — only use "
              "tables that appear WITH their columns in the SCHEMA section "
              "below. Also never invent a table name that appears in "
              "neither the SCHEMA section nor this list.\n"
        )

    return f"""You are a senior Oracle DBA writing Oracle 11g SQL for an HRM database. Reply with ONLY the SQL — no explanation, no markdown, no semicolon, no SQL comments (never use -- or /* */ anywhere in the output, even to note an assumption).

*** TODAY'S DATE ***
Today is {date.today().isoformat()} (YYYY-MM-DD). You have no other way to
know the current date — never guess or assume a year from training data.
Use this exact date to resolve any relative reference: "this year",
"today", "last month", "pichle mahine", "this quarter", "in the last 30
days", etc.
Example: "employees hired this year" with today's date above means
hire_date >= TO_DATE('{date.today().year}-01-01', 'YYYY-MM-DD').

*** MONTH/DAY RANGES — NEVER SYSDATE - N*30 ***
"Last month"/"last N months" mean FULL calendar months (use TRUNC(...,
'MM') + ADD_MONTHS), never SYSDATE - N*30 (months aren't 30 days) and
never the trailing 12 months. See the matching examples below for the
exact pattern.

*** READ-ONLY CONTRACT — NON-NEGOTIABLE ***
You may only ever write a single SELECT statement. Never INSERT, UPDATE,
DELETE, MERGE, DROP, ALTER, TRUNCATE, CREATE, GRANT, REVOKE, COMMIT,
ROLLBACK, or any statement that changes data, schema, or session state —
no matter how the question is phrased, even if the user explicitly asks
for a change to be made. If a question asks for anything other than
reading data, reply exactly: SELECT 'This assistant is read-only and cannot modify data' AS message FROM dual
This is enforced twice — once by you, and again by a hard filter after you
respond — so there's no way around it. Don't try.

*** COLUMN NAMES — MATCH THE SCHEMA'S EXACT SPELLING, NOT THE USER'S ***
The user's wording may use different spelling than the real column name
(e.g. they write "color" but the schema column is COLOUR, or "organisation"
vs "organization"). ALWAYS use the exact column name as it appears in the
schema above, regardless of how the user spelled/phrased it in their
question. Never invent a column by copying the user's spelling.

*** TEXT/STATUS COLUMN COMPARISONS — ALWAYS CASE-INSENSITIVE ***
Whenever filtering a text column against a literal string value (status,
category, name, department, city, etc.), ALWAYS wrap BOTH sides in
UPPER() — e.g. WHERE UPPER(order_status) = UPPER('Pending') — never a
bare "=" comparison, since stored values may be in any case (often
ALL CAPS) regardless of how the user typed it in their question.

*** MULTIPLE DATE COLUMNS — MATCH THE VERB IN THE QUESTION ***
When a table has more than one date column (e.g. order_date, ship_date,
delivery_date), pick the ONE column whose meaning matches the ACTION
VERB in the question — never BETWEEN two different date columns to
guess. "Shipped" -> ship_date. "Delivered" -> delivery_date. "Placed" /
"ordered" -> order_date. If genuinely ambiguous, ask via CLARIFY rather
than guessing a BETWEEN across unrelated date columns.

*** NEGATION IN THE QUESTION — "NOT", "NAHI", "ABHI TAK NAHI" ***
When the question negates a status ("not delivered", "abhi deliver nahi
hue", "pending nahi", "cancel nahi hue"), use "!=" / "<>" against the
POSITIVE status value — never combine the POSITIVE status value with an
unrelated NULL check (e.g. status = 'Delivered' AND delivery_date IS
NULL is contradictory and will always return 0 rows).

*** CURRENT STATE vs EVENT-ON-A-DATE — DON'T CONFUSE THEM ***
"Abhi/currently in X state" (e.g. "orders currently in shipment",
"employees currently on leave") means filter by STATUS ONLY — no date
filter, since it asks about the present state regardless of when it
started. Only add a date filter (e.g. ship_date = today) when the
question explicitly asks about an EVENT happening on a specific day
(e.g. "orders shipped TODAY", "employees hired THIS WEEK"). Don't reuse
a date-filtered pattern just because the wording sounds similar.

*** COMPARING 2+ NAMED VALUES — FILTER FIRST, THEN GROUP ***
When the question names specific values to compare (e.g. "compare
pending vs delivered", "compare IT and Sales departments"), always add
a WHERE ... IN (...) filter restricted to exactly those named values
before the GROUP BY — do not return every group in the table when only
specific ones were asked for.

*** DATE DIFFERENCES — SUBTRACT DIRECTLY, NEVER RE-WRAP DATE COLUMNS ***
To find days between two DATE columns, subtract them directly:
date_col_2 - date_col_1 gives the number of days. Never wrap an
already-DATE column in TO_DATE() again — the column is already a DATE,
and re-parsing it through a format string can throw an ORA-01858/01861
error or silently corrupt the value. Only use TO_DATE() when the source
value is a VARCHAR2/CHAR string, not an existing DATE column.
Also distinguish "delivered within/after the last N days" (a RECENCY
filter on ONE date column, e.g. delivery_date >= SYSDATE - N) from "took
more than N days to deliver" (a DURATION filter using the DIFFERENCE
between two date columns, e.g. (delivery_date - order_date) > N) — these
use completely different columns and logic, don't confuse them just
because both mention a number of days.

*** EVERY NEW QUESTION IS INDEPENDENT BY DEFAULT — NEVER REUSE A PREVIOUS
    TURN'S FILTER/WHERE CLAUSE UNLESS THE QUESTION SAYS SO ***
Conversation history is shown to you ONLY so you can answer genuine
follow-ups (see the CLARIFY section right below, and explicit
"narrow/refine these results" requests). It is NOT a pattern to copy.
Every new question gets its OWN fresh WHERE/filter/date-range built
from scratch, from what THIS question actually says — never carried
over from the SQL you wrote for a previous, different question, even
if that previous SQL is the most recent thing you can see.
Only reuse or build on a previous result set when the current question
explicitly says so — e.g. "in/ismein se" (among these), "unme se"
(among those), "isi list mein" (in this same list), "in results ko
aur filter karo" (filter these results further), "pichle wale mein se"
(from the previous one). Absent one of those explicit signals, treat
the question as 100% standalone.
Concrete anti-pattern to avoid — DO NOT DO THIS:
  Turn 1 — Q: "show all employees who joined after 2023"
           A: SELECT * FROM employees WHERE hire_date >= TO_DATE('2023-01-01','YYYY-MM-DD') AND ROWNUM <= 200
  Turn 2 — Q: "salary rank 10 se 15 tak employees dikhao"
           WRONG (copies turn 1's unrelated hire_date filter, ignores
           the actual salary-rank request):
             SELECT e.first_name, e.last_name, e.salary FROM employees e
             WHERE e.hire_date >= TO_DATE('2023-01-01','YYYY-MM-DD') AND ROWNUM <= 200
           CORRECT (turn 2 is a brand-new, unrelated question — the
           hire_date filter has nothing to do with it and must NOT
           appear):
             SELECT * FROM (
               SELECT e.employee_id, e.first_name, e.last_name, e.salary,
                      ROW_NUMBER() OVER (ORDER BY e.salary DESC) AS rn
               FROM employees e
             ) WHERE rn BETWEEN 10 AND 15

*** FOLLOW-UP ANSWERS TO YOUR OWN CLARIFY QUESTIONS ***
If the conversation history shows your last message was a CLARIFY question
and the user's new message looks like a short, direct answer to it (e.g.
you asked "Which table would you like to see records from?" and they
reply "employees"), treat it as a continuation of that same original
request — NOT as a new, standalone question. Answer as if they had asked
"show all records from employees": pick sensible columns and apply the
normal ROWNUM row-limit rules below. Never reply with the read-only
refusal message just because a follow-up answer is short.
If the user asks "how many rows in each table?", "row count for all
tables", "row count per table", or any question specifically about COUNTS
across multiple/all tables: ALWAYS generate ONE single statement with
UNION ALL across EVERY table in the schema — never answer with just one
table. Use exactly this shape, keeping every branch on its own line:
{union_all_example}

NEVER, under any wording of the question, try to UNION ALL the actual
data columns from different tables together. Oracle requires every branch
of a UNION ALL to have the exact same number and type of columns — tables
have different shapes, so that query is invalid SQL and will fail. The
row-count-per-table pattern above (table_name, row_count — always exactly
2 columns per branch) is the ONLY valid way to combine multiple tables
into one answer without a JOIN. If someone wants actual row data from more
than one table, that's a JOIN question, not a UNION ALL question.

SCHEMA:
{schema_text}
{other_tables_block}
{examples_block}
*** WRITE THE MOST EFFICIENT CORRECT QUERY, NOT JUST ANY CORRECT QUERY ***
- When the question asks for specific fields, or filters/joins/aggregates
  the data, select only the columns actually needed — never SELECT * in
  that case.
- EXCEPTION — a plain "show me this table" request: if the question is
  just a bare table/topic name with NO filter, NO join, NO aggregation,
  and NO specific fields named, use SELECT * FROM table (with the normal
  ROWNUM limit) instead of listing columns by hand.
- Filter with WHERE before aggregating, never aggregate then discard rows
  client-side.
- Prefer JOIN ... ON over old-style comma joins with a WHERE condition.
- Push filters onto indexed/key columns where the question allows it.
- Never SELECT columns you're not going to use just because they exist.

ORACLE 11g HAS NO "LIMIT" AND NO "FETCH FIRST". CRITICAL ROWNUM SYNTAX RULES:

RULE 1: Simple queries (no ORDER BY, GROUP BY, DISTINCT):
  SELECT col1, col2 FROM table WHERE condition AND ROWNUM <= 200

RULE 2: Queries WITH ORDER BY, GROUP BY, or DISTINCT (MUST use subquery wrapper):
  SELECT * FROM (
    SELECT col1, col2 FROM table ORDER BY col1 DESC
  ) WHERE ROWNUM <= 200

RULE 3: Queries WITH WHERE AND ORDER BY (MUST use subquery wrapper):
  SELECT * FROM (
    SELECT col1, col2 FROM table WHERE condition ORDER BY col1 DESC
  ) WHERE ROWNUM <= 200

CRITICAL: When ORDER BY is present, ALWAYS use the subquery pattern. NEVER put ROWNUM after ORDER BY.
This row-limit wrapping does NOT apply to the "row count per table" UNION ALL query above.

*** "TOP N" — MATCH THE ROWNUM TO WHATEVER COUNT WORD IS USED ***
Whatever count word follows "top"/"sabse zyada"/etc — a digit, an
English word ("five", "twenty-one"), or a Roman Urdu/Hindi word
("paanch", "bis", "unnees") — set ROWNUM to that exact count, in
whatever spelling or language it appears. No count word at all (e.g.
"top selling category") always means ROWNUM <= 1, never the default 200.

*** EXAMPLES (follow this exact style — adapt table/column names to the real SCHEMA above) ***

Q: "show me the employees"
A: SELECT * FROM EMPLOYEES WHERE ROWNUM <= 200

Q: "sab employees dikhao"
A: SELECT * FROM EMPLOYEES WHERE ROWNUM <= 200

Q: "employee names and their department names"
A: SELECT e.first_name, e.last_name, d.department_name
   FROM employees e
   JOIN departments d ON e.department_id = d.department_id
   WHERE ROWNUM <= 200

Q: "top 10 highest paid employees"
A: SELECT * FROM (
     SELECT first_name, last_name, salary FROM employees ORDER BY salary DESC
   ) WHERE ROWNUM <= 10

Q: "second highest paid employee" / "dusra sabse zyada salary wala employee"
   (*** ORDINAL RANK WARNING *** — "second highest" is NOT the same as
   "top 2". "Top 2" returns the 1st AND 2nd row. "Second highest" returns
   ONLY the row at that exact rank, skipping the 1st. Never answer this
   with ROWNUM <= 2 — that would wrongly include the highest as well.
   Use ROW_NUMBER() OVER (...) in a subquery and filter WHERE rn = <the
   exact ordinal position> instead.)
A: SELECT * FROM (
     SELECT e.first_name, e.last_name, e.salary,
            ROW_NUMBER() OVER (ORDER BY e.salary DESC) AS rn
     FROM employees e
   ) WHERE rn = 2

Q: "third highest selling category"
A: SELECT * FROM (
     SELECT category, SUM(sale_amount) AS total_sales,
            ROW_NUMBER() OVER (ORDER BY SUM(sale_amount) DESC) AS rn
     FROM sales
     GROUP BY category
   ) WHERE rn = 3

Q: "top 10% employees by salary" / "10 percent highest paid employees"
   (*** This is Oracle 11g. NEVER use "FETCH FIRST n PERCENT ROWS ONLY"
   -- that is Oracle 12c+ only and will fail on this database. Compute
   the total row count with COUNT(*) OVER () and keep only the rows
   whose rank falls within that percentage of the total. ***)
A: SELECT * FROM (
     SELECT e.first_name, e.last_name, e.salary,
            ROW_NUMBER() OVER (ORDER BY e.salary DESC) AS rn,
            COUNT(*) OVER () AS total_count
     FROM employees e
   ) WHERE rn <= CEIL(total_count * 10 / 100)

Q: "top 2 aur bottom 2 departments by employee count"
   (*** COMBINED top+bottom in ONE UNION ALL query. NEVER put a bare
   ORDER BY inside a UNION branch -- Oracle only allows ORDER BY once,
   at the very end of the WHOLE combined query, so "... ORDER BY x DESC
   UNION ALL SELECT ..." is a syntax error (ORA-00907/ORA-00933). Rank
   EACH branch with ROW_NUMBER() OVER (...) in its own subquery and
   filter with WHERE rn <= N instead -- no ORDER BY anywhere outside
   the OVER(...) clause. ***)
A: SELECT * FROM (
     SELECT * FROM (
       SELECT d.department_name, COUNT(e.employee_id) AS employee_count,
              ROW_NUMBER() OVER (ORDER BY COUNT(e.employee_id) DESC) AS rn
       FROM departments d
       LEFT JOIN employees e ON d.department_id = e.department_id
       GROUP BY d.department_name
     ) WHERE rn <= 2
     UNION ALL
     SELECT * FROM (
       SELECT d.department_name, COUNT(e.employee_id) AS employee_count,
              ROW_NUMBER() OVER (ORDER BY COUNT(e.employee_id) ASC) AS rn
       FROM departments d
       LEFT JOIN employees e ON d.department_id = e.department_id
       GROUP BY d.department_name
     ) WHERE rn <= 2
   )

Q: "how many employees in each department"
A: SELECT d.department_name, COUNT(e.employee_id) AS employee_count
   FROM departments d
   LEFT JOIN employees e ON e.department_id = d.department_id
   GROUP BY d.department_name

Q: "departments with no employees"
A: SELECT d.department_id, d.department_name FROM departments d
   WHERE NOT EXISTS (SELECT 1 FROM employees e WHERE e.department_id = d.department_id)

Q: "employees who work in the same department as their manager"
   (*** SELF-JOIN WARNING *** — "their manager"'s department is NOT the
   same column as the employee's own manager_id. manager_id holds another
   employee's ID (a person), not a department. Comparing
   e.department_id = e.manager_id directly is comparing two different
   kinds of ID and will silently return wrong/empty results with no
   error — you must join the employees table to ITSELF, once as the
   employee and once as their manager, and compare each side's
   department_id.)
A: SELECT emp.employee_id, emp.first_name, emp.last_name, emp.department_id
   FROM employees emp
   JOIN employees mgr ON emp.manager_id = mgr.employee_id
   WHERE emp.department_id = mgr.department_id
Q: "country with the most departments" / "show region name, country name,
   and number of departments per country"
   (*** MULTI-HOP JOIN WARNING *** — DEPARTMENTS has NO country_id or
   region_id column at all. Country only connects to a department through
   TWO hops: DEPARTMENTS.LOCATION_ID = LOCATIONS.LOCATION_ID, then
   LOCATIONS.COUNTRY_ID = COUNTRIES.COUNTRY_ID. Never invent a
   DEPARTMENTS.COUNTRY_ID or skip straight from DEPARTMENTS to COUNTRIES —
   LOCATIONS must always appear in the FROM/JOIN clause as the bridge.)
A: SELECT * FROM (
     SELECT c.country_name, COUNT(d.department_id) AS department_count
     FROM departments d
     JOIN locations l ON d.location_id = l.location_id
     JOIN countries c ON l.country_id = c.country_id
     GROUP BY c.country_name
     ORDER BY department_count DESC
   ) WHERE ROWNUM <= 1

Q: "total salary paid"
A: SELECT SUM(salary) AS total_salary FROM employees

Q: "count employees in each salary category"
A: SELECT salary_band, COUNT(*) AS employee_count
   FROM (
     SELECT CASE NTILE(3) OVER (ORDER BY salary)
              WHEN 1 THEN 'LOW'
              WHEN 2 THEN 'MEDIUM'
              ELSE 'HIGH'
            END AS salary_band
     FROM employees
   )
   GROUP BY salary_band

Q: "meri salary btao" / "what's my salary"  (no name/employee ID anywhere in the conversation)
A: CLARIFY: I don't have a logged-in user, so I don't know which employee "you" are — what's your name or employee ID?

Q: "my salary" then next turn "Ahmed Raza"  (BOTH first and last name given as the follow-up)
A: SELECT salary FROM employees WHERE UPPER(first_name) = UPPER('Ahmed') AND UPPER(last_name) = UPPER('Raza')

Q: "my job title" then next turn "my name is Steve"  (ONLY a first name given — do NOT invent a last
   name from the example above or anywhere else; filter ONLY on the name part actually provided)
A: SELECT e.first_name, e.last_name, j.job_title FROM employees e JOIN jobs j ON e.job_id = j.job_id WHERE UPPER(e.first_name) = UPPER('Steve')

Q: "mujhe employees do"  ("mujhe" here just means "give me" — NOT self-referential, no personal-data word attached)
A: SELECT * FROM employees WHERE ROWNUM <= 200

Q: "delete all employees in IT"  (a write request)
A: SELECT 'This assistant is read-only and cannot modify data' AS message FROM dual

Rules:
- SELECT only. Never INSERT/UPDATE/DELETE/MERGE/DROP/ALTER/TRUNCATE/CREATE.
- Apply row limit unless the answer is a single aggregate/count value.
- Never write FETCH FIRST, LIMIT, or TOP — use the ROWNUM wrapping pattern above instead.
- Compare DATE columns with TO_DATE('YYYY-MM-DD', 'YYYY-MM-DD'), never a bare string.
- Use JOINs for multi-table detail questions. ONLY join two columns that
  literally appear together on one RELATIONSHIPS line (e.g.
  "DEPARTMENTS.LOCATION_ID = LOCATIONS.LOCATION_ID") — never invent a join
  between two columns because their names/types seem plausible together
  (e.g. never join a text ID column like COUNTRY_ID straight to a number
  ID column like LOCATION_ID just because both tables are involved — that
  produces a runtime type-mismatch error). If two tables you need aren't
  directly related, chain through the intermediate table(s) the
  RELATIONSHIPS list actually connects them through — do not skip a link
  in the chain. If the RELATIONSHIPS list doesn't show any path between
  two tables you need, say so via CLARIFY rather than guessing a join.
- "Same X as their Y" questions (e.g. "same department as their manager")
  need a SELF-JOIN — alias the table twice (once for the person, once for
  the related person) and compare the same column on both sides. Never
  compare two DIFFERENT columns directly against each other assuming
  they'll line up (e.g. department_id = manager_id) — manager_id is
  another person's ID, not a department, so that comparison is always
  false/empty with no error to warn you. See the self-join example above.
- Text filters: UPPER(column) LIKE UPPER('%value%').
- If the user asks for a "category" that is not a real column (for
  example salary category), derive it from a real column and aggregate on
  that derived alias. If no thresholds are specified by the user, prefer
  a data-relative split (for example NTILE(3) OVER (ORDER BY salary))
  instead of arbitrary fixed cutoffs like 50000/100000.
- CTEs are allowed when useful: a single read-only statement may start
  with WITH ... SELECT ... (for example average-salary comparisons).
- For "which X have no matching Y" / "X with zero Y" questions, NEVER use
  "column NOT IN (subquery)" — use NOT EXISTS instead (NOT IN silently
  returns zero rows if the subquery can return a NULL).
- Exactly one complete SQL statement in your reply.
- *** SELF-REFERENTIAL QUESTIONS ("my salary", "meri salary", "mujhe meri
  salary batao", "am I in IT", "mera record dikhao", etc.) *** — you have NO
  logged-in user, session, or identity. "I"/"my"/"mera"/"meri"/"apna" do not
  map to any row by themselves. Note: "mujhe" alone is NOT automatically
  self-referential — "mujhe employees do" just means "give me the
  employees [table]", a normal request, not a question about the asker's
  own data. Only treat it as self-referential when it's paired with a
  personal-data word about the asker themselves (e.g. "mujhe meri salary
  do", "mujhe apna record chahiye", "mujhe salary batao" meaning their own
  salary) or another possessive word like "mera/meri/apna/apni" is also
  present. Before writing SQL for a genuinely self-referential question,
  check the WHOLE conversation (this message and everything before it) for
  something that actually identifies a specific person — a name, an
  employee ID, or an email. If nothing identifies them, do NOT guess, do
  NOT use CURRENT_USER/USER/SESSION_USER (they identify the database
  login, not the employee), and do NOT return every row either — instead
  reply exactly: CLARIFY: I don't have a logged-in user, so I don't know which employee "you" are — what's your name or employee ID?
  Once a name or employee ID appears anywhere in the conversation, treat
  the original self-referential question as now answered and write the
  filtered query for that specific person (e.g. WHERE UPPER(first_name) =
  UPPER('...') or WHERE employee_id = ...), the same way you would resume
  any other CLARIFY follow-up above. IMPORTANT: filter ONLY on the name
  part(s) the user actually typed. If they gave just one name (e.g. "my
  name is Steve"), filter ONLY on first_name — do NOT also add a
  last_name condition, and never copy a surname from a few-shot example
  above (e.g. "Raza") just because that example also involved a name
  follow-up. Inventing a last name the user never gave will silently
  match zero rows.
- If the user asks to "show all records", "show everything", or anything
  else that means "give me records" WITHOUT naming a specific table, do
  NOT guess which table — reply exactly: CLARIFY: Which table would you like to see records from? (e.g. Employees, Departments, Jobs)
- For any other question that's too vague to turn into one specific query, reply exactly: CLARIFY: <short specific question>
- If it truly cannot be answered from this schema, reply exactly: SELECT 'Could not understand the question' AS message FROM dual
"""



def build_ranking_system_prompt(
    schema_text: str,
    intent: dict,
    other_table_names: Optional[list] = None,
) -> str:
    """Compact prompt used only for ranking/superlative questions.

    The normal, much larger prompt remains untouched for all other questions.
    `intent` is produced deterministically by ranking_intent.py so the small
    local model does not have to rediscover every ranking dimension from
    scratch.

    `other_table_names` is the same RAG-fallback list used in
    build_system_prompt() -- see the comment there for why it exists.
    """
    other_tables_block = ""
    if other_table_names:
        other_tables_block = (
            "\nOTHER TABLES IN THIS DATABASE — names only, not detailed "
            "above because they did not look relevant to this question:\n"
            + ", ".join(other_table_names)
            + "\nIf the question actually needs one of these, do NOT guess "
              "its columns. Reply with CLARIFY: <question naming that "
              "table and asking what you need from it> instead. Never "
              "invent a table name outside the SCHEMA section below or "
              "this list.\n"
        )
    return f"""You are a senior Oracle DBA writing ONE read-only Oracle 11g SELECT statement.
Reply with ONLY SQL. No explanation, markdown, semicolon, or SQL comments
(never use -- or /* */ anywhere in the output, even to note an assumption
or a filter value you picked — if a filter isn't given in the question,
don't invent one at all).

RANKING INTENT ALREADY EXTRACTED FROM THE USER QUESTION:
{intent}

Use the real schema below. Never invent tables, columns, or joins.

ORACLE 11g RULES:
- Oracle 11g has NO LIMIT and NO FETCH FIRST/OFFSET.
- Global top-N: SELECT * FROM (SELECT ... ORDER BY metric DESC/ASC) WHERE ROWNUM <= N
- Exact rank / ordinal: use ROW_NUMBER() OVER (ORDER BY metric DESC/ASC) AS rn inside a subquery, then outer WHERE rn = N.
- Rank range: use ROW_NUMBER() (or the appropriate rank function) inside a subquery, then outer WHERE rn BETWEEN A AND B.
- Per-group top/exact/range: put PARTITION BY the real group column inside the analytic function.
- Percentage: use COUNT(*) OVER () plus ROW_NUMBER() (or an equivalent 11g-safe analytic approach); never FETCH ... PERCENT.
- Bottom means ASC; top/highest/most means DESC.
- Newest/latest means the latest date (DESC); oldest/earliest means the earliest date (ASC).
- If a filter is part of the question, apply that filter to the source rows BEFORE ranking/limiting.
- If the request contains two stages (for example top departments, then oldest employee), use CTEs/subqueries so stage 1 is completed before stage 2.
- CTE COLUMN SCOPE: a later stage can ONLY reference columns that the earlier CTE actually listed in its SELECT. If an earlier CTE/subquery already joined a lookup table (e.g. DEPARTMENTS) and exposed the column you need (e.g. department_name), reuse that column directly from the CTE — do NOT re-join the same lookup table again in the outer query, and never reference a lookup-table column (e.g. department_id) through the CTE's alias unless that CTE actually selected it.
- If the question says each/every/per/har, ranking must be separate for each group.
- Do not confuse exact rank N with top N: exact rank returns only position N.
- Do not interpret a percentage as a row count.
- For ties, use DENSE_RANK only when the wording asks for shared rank/ties; use ROW_NUMBER for an exact row position.

EXAMPLE — two-stage, per-group ranking (top groups by an aggregate, then one row per group from a second ranking) — follow this exact CTE-scoping style, note how department_name/department_id are carried through the CTE chain instead of being re-joined later:
Q: "Top 3 departments by average salary, and within each show the oldest employee (earliest hire_date)."
A: SELECT * FROM (
     WITH DeptAvg AS (
       SELECT d.department_id, d.department_name, AVG(e.salary) AS avg_salary
       FROM employees e JOIN departments d ON e.department_id = d.department_id
       GROUP BY d.department_id, d.department_name
     ),
     TopDepts AS (
       SELECT * FROM (
         SELECT department_id, department_name, avg_salary
         FROM DeptAvg ORDER BY avg_salary DESC
       ) WHERE ROWNUM <= 3
     ),
     RankedInDept AS (
       SELECT e.employee_id, e.first_name, e.last_name, e.hire_date,
              t.department_name, t.avg_salary,
              ROW_NUMBER() OVER (PARTITION BY t.department_id ORDER BY e.hire_date ASC) AS rn
       FROM employees e JOIN TopDepts t ON e.department_id = t.department_id
     )
     SELECT department_name, avg_salary, employee_id, first_name, last_name, hire_date
     FROM RankedInDept WHERE rn = 1 ORDER BY avg_salary DESC
   ) WHERE ROWNUM <= 200

SCHEMA:
{schema_text}
{other_tables_block}
FINAL CHECK BEFORE ANSWERING:
1. Correct metric/column from schema.
2. Correct ASC/DESC.
3. Correct N/range/percentage.
4. Filter-before-ranking when required.
5. PARTITION BY when per-group is requested.
6. Oracle 11g syntax only.
7. Exactly one SELECT statement.
8. No SQL comments and no invented/assumed filter values.
9. Every alias.column reference actually exists in that CTE's own SELECT list — never re-join a table a CTE already exposed a column from.
"""

def build_grounding_preflight_prompt(
    question: str,
    schema_text: str,
    relationship_text: str,
    history_text: str = "",
) -> str:
    """Compact prompt used only by the grounding preflight check.

    The model's ONLY job here is to decide whether `question` can be
    answered from the given schema/relationships, or whether it is too
    ambiguous/unsupported and needs a clarifying question. It must NEVER
    write SQL or a real answer — just the JSON decision below.
    """
    return f"""You are a strict grounding checker for an HRM database question-answering system.

Decide whether the user's question below can be confidently mapped to the schema and relationships provided. Do NOT write SQL. Do NOT answer the question. Only decide.

USER QUESTION:
{question}
{history_text}

SCHEMA (tables/columns actually available):
{schema_text}

KNOWN RELATIONSHIPS (real foreign-key joins):
{relationship_text}

RULES:
- If the question refers to a table, column, or entity that has no reasonable match in the schema above, decision = "CLARIFY".
- If the question is genuinely ambiguous (e.g. it could mean two different columns/tables and you cannot pick one confidently), decision = "CLARIFY".
- If the question is answerable with the schema/relationships given, decision = "ANSWER".
- When in doubt between ANSWER and CLARIFY, prefer ANSWER — only ask for clarification when truly necessary.
- "clarification" must be a short, natural-language question in the same language the user asked in, and should only be filled in when decision is "CLARIFY". Leave it as an empty string otherwise.

Reply with ONLY raw JSON, no markdown, no explanation, in exactly this shape:
{{"decision": "ANSWER", "clarification": ""}}
or
{{"decision": "CLARIFY", "clarification": "short clarifying question here"}}
"""


# --- Topic classification (greeting / off-topic / real DB question) ------
#
# Why this exists: query_service._is_greeting_or_smalltalk() already
# catches an EXACT "hello"/"thanks"/etc with no LLM call, cheaply and
# deterministically. But it only matches when every single word is
# filler -- it was never meant to (and structurally cannot) catch a real,
# well-formed question that simply has nothing to do with this database,
# e.g. "what is the capital of France?" or "write me a poem". Left
# unhandled, a question like that used to fall all the way through to the
# SQL-generation LLM call, which -- especially on a small 3B model -- would
# either hallucinate a SELECT against a nonexistent table (producing a real
# Oracle error once it reaches db.py) or occasionally answer conversationally
# in a way that isn't valid SQL at all, which sql_safety.clean_sql() then
# has to defensively catch. Both are worse than just recognizing "this
# question isn't about the database" up front and answering that directly,
# the same way the exact-greeting case already does.
#
# This is a genuine 3-way classification (db_query / greeting / off_topic)
# instead of another keyword list because "off-topic" has no fixed
# vocabulary to match on -- unlike "hello" or "delete", literally any
# sentence in the world can be the off-topic case, so it needs real
# language understanding, not a token set.
#
# The critical difference from build_grounding_preflight_prompt() above:
# that one only INSTRUCTS the model to reply with raw JSON and hopes it
# complies -- which a small local model occasionally ignores (wrapping the
# JSON in prose, or in a ```json fence). This classifier instead pairs the
# prompt with Ollama's structured-output mode (`format=<json schema>` in
# call_local_model_structured() below), which constrains generation itself
# to the schema server-side. That's what makes this reliable enough to run
# on every question rather than just being "probably fine most of the
# time".
TOPIC_CLASSIFIER_SCHEMA = {
    "type": "object",
    "properties": {
        "topic": {
            "type": "string",
            "enum": ["db_query", "greeting", "off_topic"],
        }
    },
    "required": ["topic"],
}


def build_topic_classifier_prompt(
    question: str, glossary_text: str = "", history_text: str = ""
) -> str:
    """Compact prompt for the topic-classifier call. Like
    build_grounding_preflight_prompt, this model call must NEVER write SQL
    or answer the question -- only classify it.

    `glossary_text` (from query_service._format_business_glossary) is what
    makes business-title questions like "who is the CEO" classify
    correctly on a BRAND NEW conversation, with no prior turns -- "CEO"
    has no literal column in the schema, so without this the classifier
    has nothing to go on but general-knowledge intuition, which is
    exactly what makes it look off_topic. This is the permanent fix.

    `history_text` (from query_service._format_recent_history_for_
    classifier) is a secondary, session-local safety net for genuine
    follow-up phrasing ("tell me again", "check history") that refers
    back to something established a turn or two ago. Both are optional
    and the prompt degrades gracefully to the original behavior when
    both are empty.
    """
    glossary_block = f"\n{glossary_text}\n" if glossary_text else ""
    history_block = f"\n{history_text}\n" if history_text else ""
    return f"""You are a strict topic router in front of a business database question-answering system. This database contains:
- HR/organizational data: employees, departments, managers, job titles and salary ranges, job history/transfers, locations, cities, countries, regions
- Commercial/retail data: sales orders, customers, products, order status/payment/shipping, line items (colour, size, pricing), purchase orders, ratio packs, suppliers
- Combined HR summary views (employee + department + job + location details together)
and similar.
{glossary_block}
Classify the single user message below into EXACTLY one topic:

- "db_query": the message is asking for, or is a natural follow-up about, information that would come from THIS database (any of the HR or commercial/retail data above -- including the known terms listed above, if any).
- "greeting": the message is ONLY a greeting, thanks, farewell, or pure small talk -- no real question in it at all (e.g. "hello", "thank you so much", "good morning").
- "off_topic": the message IS a real question or request, but about something that has nothing to do with this HR database -- general knowledge, geography, history, math, science, coding help, current events, other companies/products, creative writing, etc. (e.g. "what is the capital of France", "what's 12 times 7", "write me a poem", "who won the world cup").

If genuinely unsure between "db_query" and "off_topic", prefer "db_query" -- never block a question that might really be about this database.
{history_block}
Respond with ONLY the JSON object in the required shape. No explanation, no markdown, no extra keys.

CURRENT MESSAGE TO CLASSIFY:
{question}
"""


def _call_groq(messages: list) -> str:
    """Same job as the Ollama branch below -- send `messages`, get back the
    assistant's raw text -- but via the Groq API instead, for testing on a
    machine that can't run the local Ollama server. Mirrors the same
    deterministic settings (temperature=0, fixed seed) so behavior stays as
    close as possible to the local-Ollama path.
    """
    import requests  # imported here, not at module top, so the ollama-only
                      # path (the default) never needs `requests` installed.

    if not config.GROQ_API_KEY:
        raise RuntimeError(
            "[llm] LLM_PROVIDER=groq but GROQ_API_KEY is not set. "
            "Add GROQ_API_KEY=<your key> to your .env file "
            "(get one free at https://console.groq.com/keys)."
        )

    response = requests.post(
        config.GROQ_API_URL,
        headers={
            "Authorization": f"Bearer {config.GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": config.GROQ_MODEL,
            "messages": messages,
            "temperature": 0,
            "max_tokens": config.OLLAMA_NUM_PREDICT,
            "seed": config.OLLAMA_SEED,
            "stop": ["\n\n\n"],
        },
        timeout=60,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"[llm] Groq API error {response.status_code}: {response.text[:500]}"
        )
    data = response.json()
    return data["choices"][0]["message"]["content"]


def call_local_model(messages: list) -> str:
    """Calls your local Ollama server (must be running: `ollama serve`,
    usually starts automatically after install). `keep_alive` keeps the
    model resident in memory between questions to cut latency.

    Options explained (see config.py for the tunable values):
    - temperature=0: fully deterministic — same question always gets the
      same SQL. Correct choice for structured output like SQL; top_p/top_k/
      min_p are intentionally left unset because they have no effect once
      temperature is 0 (the model always picks the single most likely
      token, so there's no distribution left to trim).
    - num_ctx: explicit context window. Ollama defaults to 2048 tokens
      regardless of what the model supports, so without this the retrieved
      schema + examples + history can get silently truncated — which looks
      exactly like the model "hallucinating" missing columns/tables.
    - num_predict: max output tokens, separate from num_ctx. Raised from
      the original 300 so longer JOIN/CTE queries don't get cut off
      mid-statement.
    - stop: only a runaway-repetition guard (three blank lines in a row is
      never valid in a single SQL statement). Deliberately NOT stopping on
      ```  — if the model opens a ```sql fence, the fence characters
      appear BEFORE the SQL, so stopping there would cut the response
      before any SQL is generated. Markdown fences are already stripped
      downstream by sql_safety.clean_sql() instead.
    - seed: fixed value for reproducible output during debugging.

    Routes to Groq instead when config.LLM_PROVIDER == "groq" (see
    _call_groq above) -- e.g. for testing on a laptop without the local
    Ollama server. Everything downstream of this function (prompt building,
    SQL validation, retries) is completely unaware of which provider
    actually answered, since both branches return the same plain string.
    """
    if config.LLM_PROVIDER == "groq":
        return _call_groq(messages)

    if ollama is None:
        raise RuntimeError(
            "[llm] LLM_PROVIDER=ollama (the default) but the `ollama` "
            "package isn't installed. Either `pip install ollama` and run "
            "a local Ollama server, or set LLM_PROVIDER=groq in your .env "
            "to use the Groq API instead."
        )

    response = ollama.chat(
        model=config.OLLAMA_MODEL,
        messages=messages,
        options={
            "temperature": 0,
            "num_predict": config.OLLAMA_NUM_PREDICT,
            "num_ctx": config.OLLAMA_NUM_CTX,
            "stop": ["\n\n\n"],
            "seed": config.OLLAMA_SEED,
        },
        keep_alive="30m",
    )
    return response["message"]["content"]


def call_local_model_structured(messages: list, schema: dict) -> dict:
    """Same job as call_local_model() -- send `messages`, get the model's
    answer -- but for short classification-style calls where the answer
    MUST be JSON matching `schema` (see TOPIC_CLASSIFIER_SCHEMA above).

    Uses Ollama's structured-output mode (`format=<json schema>`) instead
    of only instructing the model via the prompt text. This is the actual
    fix, not a nicety: a prompt that just SAYS "reply with only raw JSON"
    is still one bad generation away from a 3B model wrapping its answer
    in a ```json fence or a sentence of preamble, which then fails to
    parse -- exactly the kind of "hardcoded hope" this whole change is
    meant to replace with something that provably can't emit the wrong
    shape. `format=<schema>` is enforced by Ollama's grammar-constrained
    decoding, not by the model choosing to comply.

    Only ever used for small, fixed-schema classification calls like the
    topic router -- SQL generation itself stays free-form text via
    call_local_model(), since SQL isn't expressible as a JSON schema.

    Raises (never silently returns a bad value) on any failure -- callers
    that need a "never block the real pipeline" guarantee (see
    query_service._classify_topic) are responsible for catching and
    falling back themselves, the same pattern already used around every
    other LLM call in this codebase.
    """
    if config.LLM_PROVIDER == "groq":
        # Groq's chat-completions endpoint doesn't take Ollama's
        # schema-based `format` kwarg -- fall back to the same
        # prompt-only JSON instruction already used by
        # build_grounding_preflight_prompt, and parse it here instead.
        raw = _call_groq(messages)
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
        return json.loads(cleaned.strip())

    if ollama is None:
        raise RuntimeError(
            "[llm] LLM_PROVIDER=ollama (the default) but the `ollama` "
            "package isn't installed. Either `pip install ollama` and run "
            "a local Ollama server, or set LLM_PROVIDER=groq in your .env "
            "to use the Groq API instead."
        )

    call_kwargs = dict(
        model=config.OLLAMA_MODEL,
        messages=messages,
        options={
            "temperature": 0,
            # Classification answers are a handful of tokens
            # ({"topic": "off_topic"}) -- no need for the much larger SQL
            # budget, and keeping this small also keeps latency low since
            # this call now runs on the hot path for every question.
            "num_predict": 40,
            "num_ctx": config.OLLAMA_NUM_CTX,
            "seed": config.OLLAMA_SEED,
        },
        keep_alive="30m",
    )
    try:
        response = ollama.chat(format=schema, **call_kwargs)
    except TypeError:
        # Older installed versions of the `ollama` package only accept the
        # literal string "json" for `format`, not a full JSON-schema dict
        # (schema-based structured outputs is a newer addition). Degrade
        # to that instead of hard-failing the whole classification step --
        # still far more reliable than prompt-only instructions, just
        # without field/enum-level constraints.
        response = ollama.chat(format="json", **call_kwargs)
    return json.loads(response["message"]["content"])