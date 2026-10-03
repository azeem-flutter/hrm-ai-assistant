# Claim Verification + JOIN Grounding Changes

## What changed

### 1. New `query_grounding.py`
Adds two independent protections:

- **Claim verification mode**: detects factual/superlative claims such as
  `Mobile category has the highest sales`, `John is the highest paid employee`,
  and direct comparisons such as `IT has more employees than HR`.
- **JOIN trust boundary**: validates explicit SQL JOINs against real Oracle FK
  relationships and can request one repair instead of executing an invented JOIN.
- **Clarification preflight**: relationship-heavy ambiguous questions can return
  `CLARIFY:` before SQL generation.

### 2. `query_service.py`
- Claim verification runs **before deterministic ranking**, so a claim cannot
  accidentally take the normal TOP/EXACT rank path.
- The claim prompt explicitly treats the user's assertion as untrusted.
- For a superlative claim, the model must calculate the winner independently
  (e.g. all categories' sales) and only then compare the claimed value.
- Claim SQL also passes through FK JOIN validation.
- Existing normal SQL/ranking/RAG/repair flow remains the fallback.
- Final normal SQL now gets the same FK JOIN validation before it can be returned.

### 3. `config.py`
Two opt-out switches were added:

```text
ENABLE_CLAIM_VERIFICATION=true
ENABLE_QUERY_PREFLIGHT=true
```

Set either to `false` temporarily if you need a rollback/debug comparison.

## What is intentionally NOT changed

- `app.py` and its `CLARIFY:` handling
- ranking intent categories and deterministic ranking templates
- RAG retrieval
- Oracle schema/FK discovery
- SQL safety/read-only checks
- existing ranking correction/repair logic
- conversation history behavior

## Important semantic difference

`Show sales for Mobile category` is still a normal retrieval request and can use
Mobile as a filter.

`Mobile category has the highest sales` is treated as a **claim**. Mobile is not
allowed to become the filter that determines the winner. The query must calculate
the actual winner independently and compare it with Mobile.

## Verification performed here

- All project Python files compile successfully with `py_compile`.
- Static claim detector checks passed for representative English/Hinglish cases.
- The generated ZIP contains the original project modules plus the new grounding
  module and configuration/query-service changes.

A live Oracle/Ollama run was not possible in this build environment, so the final
semantic behavior should still be tested against your actual database/model.
