from core import conversation_context


def test_self_contained_question_is_new():
    d = conversation_context.resolve_question("show top 5 employees by salary", [], enabled=True)
    assert d.action == "NEW"
    assert d.effective_question == "show top 5 employees by salary"


def test_disabled_resolver_passes_question_through():
    d = conversation_context.resolve_question("and their departments?", [], enabled=False)
    assert d.action == "NEW"
