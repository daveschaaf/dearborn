import json

from dearborn.transcript_parser import TranscriptClient, parse_question_answer_pairs


def standard_question_and_answer_segments():
    return [
        {"speaker": "Operator", "text": "Welcome to the earnings call."},
        {"speaker": "Chief Executive", "text": "Prepared remarks."},
        {
            "speaker": "Investor Relations",
            "text": (
                "We will now begin the question and answer session. The first "
                "question will come from Alice Analyst with Example Securities."
            ),
        },
        {"speaker": "Alice Analyst", "text": "What changed this quarter?"},
        {"speaker": "Chief Executive", "text": "Revenue increased."},
        {"speaker": "Chief Financial Officer", "text": "Margins expanded."},
        {"speaker": "Investor Relations", "text": "Please take the next question."},
        {
            "speaker": "Operator",
            "text": "Our next question comes from Bob Research of Sample Bank.",
        },
        {"speaker": "Bob Research", "text": "What is the outlook?"},
        {"speaker": "Chief Executive", "text": "We reaffirm guidance."},
    ]


def test_parse_question_answer_pairs_is_deterministic_and_preserves_turns():
    structured_content = standard_question_and_answer_segments()

    first_result = parse_question_answer_pairs(structured_content)
    second_result = parse_question_answer_pairs(structured_content)

    assert first_result == second_result
    assert [(pair.analyst, pair.question, pair.answer) for pair in first_result.pairs] == [
        (
            "Alice Analyst",
            "What changed this quarter?",
            "Revenue increased. Margins expanded.",
        ),
        ("Bob Research", "What is the outlook?", "We reaffirm guidance."),
    ]
    assert first_result.pairs[0].answer_speakers == (
        "Chief Executive",
        "Chief Financial Officer",
    )
    assert first_result.pairs[0].question_segment_indexes == (3,)
    assert first_result.pairs[0].answer_segment_indexes == (4, 5)
    assert not first_result.review_reasons


def test_parser_marks_missing_answer_for_review_without_fabricating_a_pair():
    result = parse_question_answer_pairs(
        [
            {
                "speaker": "Operator",
                "text": "The first question will come from Alice Analyst with Example.",
            },
            {"speaker": "Alice Analyst", "text": "What changed this quarter?"},
        ]
    )

    assert not result.pairs
    assert [review.code for review in result.review_reasons] == [
        "q_and_a_boundary_inferred_from_announcement",
        "missing_answer",
    ]


def test_parser_marks_missing_question_and_absent_q_and_a_section():
    missing_question = parse_question_answer_pairs(
        [
            {
                "speaker": "Operator",
                "text": "The first question will come from Alice Analyst with Example.",
            },
            {"speaker": "Chief Executive", "text": "An answer without a question."},
        ]
    )
    no_question_and_answer = parse_question_answer_pairs(
        [{"speaker": "Chief Executive", "text": "Prepared remarks only."}]
    )

    assert [review.code for review in missing_question.review_reasons] == [
        "q_and_a_boundary_inferred_from_announcement",
        "missing_question",
    ]
    assert [review.code for review in no_question_and_answer.review_reasons] == [
        "no_q_and_a_section"
    ]


def test_transcript_client_persists_raw_json_and_reuses_it(tmp_path, monkeypatch):
    transcript = {"structured_content": standard_question_and_answer_segments()}
    calls = []

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"rows": [{"row": transcript}]}

    def fake_get(*args, **kwargs):
        calls.append((args, kwargs))
        return Response()

    monkeypatch.setattr("dearborn.transcript_parser.requests.get", fake_get)
    client = TranscriptClient(raw_directory=tmp_path)

    assert client.load_transcript("exm", 2025, "Q1") == transcript
    raw_path = tmp_path / "EXM_2025_Q1.json"
    assert json.loads(raw_path.read_text(encoding="utf-8")) == transcript

    assert client.load_transcript("EXM", 2025, "Q1") == transcript
    assert len(calls) == 1
