"""Anchor confidence and full-quote parsing.

Two silent failure modes are guarded here: a fragment whose opening was
paraphrased used to anchor wherever the next matched word landed with no
trace, and multi-line quotes lost everything after their first line — both
weakened the alignment without anyone knowing.
"""

from clipmaker.align import _fragment_quote, anchor_starts


def segs(*words_at):
    """One transcript segment with word-level timestamps."""
    words = [{"word": w, "start": float(t), "end": float(t) + 0.4}
             for w, t in words_at]
    return [{"start": words[0]["start"], "end": words[-1]["end"] + 0.4,
             "text": " ".join(w["word"] for w in words), "words": words}]


# --- quote parsing ---------------------------------------------------------

def test_quote_takes_every_leading_blockquote_line():
    block = ("> Первый абзац фрагмента.\n"
             "> Второй абзац того же фрагмента.\n"
             "\n"
             "**Search query:** `x`\n")
    assert _fragment_quote(block) == \
        "Первый абзац фрагмента. Второй абзац того же фрагмента."


def test_unconfirmed_note_deeper_in_the_block_is_not_part_of_the_quote():
    block = ("> Настоящий текст фрагмента.\n"
             "\n"
             "### My videos\n"
             "\n"
             "> **Unconfirmed** — nothing here mentions X.\n")
    assert _fragment_quote(block) == "Настоящий текст фрагмента."


def test_hard_wrapped_quote_keeps_its_lazy_continuation_lines():
    # Storyboards written from a hard-wrapped text.md: only the first line has ">".
    block = ("> line one of narration\n"
             "line two of the same paragraph\n"
             "line three\n"
             "\n"
             "**Search query:** `x`\n")
    assert _fragment_quote(block) == "line one of narration line two of the same paragraph line three"


def test_hard_wrapped_paragraph_becomes_one_fragment_line():
    from clipmaker.youtube_search import split_into_paragraphs
    text = "word " * 25 + "\n" + "word " * 25 + "\n\n" + "other " * 45
    first = split_into_paragraphs(text, min_words=40)[0]
    assert "\n" not in first
    assert len(first.split()) == 50


# --- anchor confidence -----------------------------------------------------

def test_exact_anchor_with_full_match_scores_one():
    transcript = segs(("we", 0), ("begin", 1), ("the", 2), ("story", 3),
                      ("second", 10), ("part", 11), ("starts", 12), ("here", 13))
    starts, stats = anchor_starts(
        ["we begin the story", "second part starts here"], transcript)
    assert starts == [0.0, 10.0]
    assert stats[0] == {"match_ratio": 1.0, "anchor": "exact"}
    assert stats[1] == {"match_ratio": 1.0, "anchor": "exact"}


def test_paraphrased_opening_is_labeled_and_ratio_drops():
    # Narrator replaced the fragment's first words; only the tail matches.
    transcript = segs(("intro", 0), ("words", 1), ("here", 2),
                      ("totally", 10), ("different", 11), ("opening", 12),
                      ("matching", 13), ("tail", 14))
    starts, stats = anchor_starts(
        ["intro words here", "unspoken start matching tail"], transcript)
    assert stats[1]["anchor"] == "paraphrased"
    assert stats[1]["match_ratio"] == 0.5  # 2 of 4 words matched
    assert starts[1] == 13.0  # snapped to the first matched word


def test_fully_unvoiced_fragment_is_flagged_unmatched():
    transcript = segs(("only", 0), ("first", 1), ("fragment", 2), ("spoken", 3))
    _, stats = anchor_starts(
        ["only first fragment spoken", "чистый невойсованный хвост абзаца"],
        transcript)
    assert stats[1]["anchor"] == "unmatched"
    assert stats[1]["match_ratio"] == 0.0
