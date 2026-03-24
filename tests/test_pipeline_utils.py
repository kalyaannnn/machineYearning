from pipeline_utils import promotion_gate, score_prompt_output


def test_score_prompt_output_json_format():
    prompt = {
        "instruction": "Return valid JSON with keys topic and summary.",
        "expected_format": "json",
        "must_include": ["topic", "summary"],
    }
    out = '{"topic":"optim","summary":"short note"}'
    score = score_prompt_output(prompt, out)
    assert score["format"] == 1.0
    assert score["must_include"] == 1.0
    assert 0.0 <= score["coherence"] <= 1.0


def test_promotion_gate_passes_on_gain_and_no_regression():
    pre = {
        "coherence_score": 0.40,
        "code_token_acc": 0.30,
        "math_exact_match": 0.20,
    }
    sft = {
        "coherence_score": 0.46,
        "code_token_acc": 0.31,
        "math_exact_match": 0.191,
    }
    gate = promotion_gate(pre, sft)
    assert gate["passed"] is True
    assert gate["checks"]["coherence_improved"] is True
    assert gate["checks"]["code_not_regressed"] is True
    assert gate["checks"]["math_not_regressed"] is True


def test_promotion_gate_fails_on_large_regression():
    pre = {
        "coherence_score": 0.50,
        "code_token_acc": 0.40,
        "math_exact_match": 0.35,
    }
    sft = {
        "coherence_score": 0.54,
        "code_token_acc": 0.37,
        "math_exact_match": 0.34,
    }
    gate = promotion_gate(pre, sft)
    assert gate["passed"] is False
    assert gate["checks"]["coherence_improved"] is True
    assert gate["checks"]["code_not_regressed"] is False
