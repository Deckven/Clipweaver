"""Retrieval eval: the metric, the lexical layers, gold building and review round-trip.

The metric is what the per-layer verdicts rest on, so the edge cases that would
silently inflate or deflate recall are pinned here: one video returned ten times,
drifting scene times, whole-video hits with no time, typographic apostrophes.
"""

from clipmaker.retrieval_eval import (
    Bm25Layer,
    Moment,
    Pick,
    attribution,
    evaluate_item,
    file_rank,
    moment_rank,
    name_key,
    ranked_moments,
    recall_at_k,
    summarize,
    tokenize,
)
from clipmaker.retrieval_gold import (
    apply_review,
    clean_label,
    items_from_events,
    items_from_shots,
    merge_reviews,
    render_sheet,
)

A = "Assassin's Creed IV Black Flag Resynced Ship Upgrade Trading 1.mp4"
B = "Assassin's Creed IV Black Flag Resynced Battle Jackdaw 1.mp4"
C = "Eve Online Orca flies 1.mp4"


def m(name, start, end=None, score=1.0):
    return Moment(name, start, end, score)


# ---------------------------------------------------------------- metric

def test_moment_rank_finds_first_overlap():
    pick = Pick(A, 20.0, 30.0)
    moments = [m(B, 20, 25), m(A, 100, 105), m(A, 28, 33)]
    assert moment_rank(moments, pick) == 3


def test_moment_rank_honours_tolerance_for_drifting_scene_times():
    pick = Pick(A, 20.0, 30.0)
    assert moment_rank([m(A, 33, 38)], pick, tol=5) == 1
    assert moment_rank([m(A, 36, 40)], pick, tol=5) is None


def test_whole_video_hit_counts_for_file_not_moment():
    pick = Pick(A, 20.0, 30.0)
    moments = [m(A, None)]
    assert moment_rank(moments, pick) is None
    assert file_rank(moments, pick) == 1


def test_file_rank_dedupes_repeated_videos():
    pick = Pick(C, 0, 5)
    moments = [m(A, 0, 5), m(A, 10, 15), m(A, 20, 25), m(B, 0, 5), m(C, 300, 305)]
    assert file_rank(moments, pick) == 3
    assert moment_rank(moments, pick) is None


def test_backwards_scene_interval_gets_a_default_span():
    pick = Pick(A, 20.0, 30.0)
    assert moment_rank([m(A, 22, 10)], pick, tol=0) == 1


def test_name_key_folds_typographic_apostrophe_path_and_trademark():
    path = "D:\\footage\\Assassin’s Creed IV Black Flag Resynced\\" + A
    assert name_key(path) == name_key(A)
    assert name_key("Assassin's Creed™ Black Flag Resynced_World_Map.mp4") == \
        name_key("assassin's creed black flag resynced_world_map.mp4")


def test_recall_at_k_counts_ranks_within_k():
    ranks = [1, 3, None, 7]
    assert recall_at_k(ranks, 1) == 0.25
    assert recall_at_k(ranks, 5) == 0.5
    assert recall_at_k(ranks, 10) == 0.75
    assert recall_at_k([], 5) == 0.0


class FakeLayer:
    def __init__(self, answers):
        self.answers = answers

    def search(self, query, n):
        return self.answers.get(query, [])[:n]


def test_evaluate_item_takes_best_query_and_best_pick():
    item = {"queries": ["q1", "q2"],
            "picks": [{"file": A, "start": 20, "end": 30, "in_catalog": True},
                      {"file": C, "start": 0, "end": 5, "in_catalog": False}]}
    layer = FakeLayer({"q1": [m(B, 0, 5), m(A, 200, 205)],
                       "q2": [m(A, 25, 30)],
                       # out-of-catalog pick must not count even if "returned"
                       })
    ranks = evaluate_item(item, {"semantic": layer})["semantic"]
    assert ranks == {"file": 1, "moment": 1, "query": "q2"}


def test_summary_any_layer_and_attribution():
    rows = [
        {"id": "a", "video": "v1", "ranks": {"semantic": {"file": 1, "moment": 2},
                                             "ocr": {"file": None, "moment": None}}},
        {"id": "b", "video": "v1", "ranks": {"semantic": {"file": None, "moment": None},
                                             "ocr": {"file": 4, "moment": 9}}},
    ]
    table = summarize(rows, ("semantic", "ocr"), ks=(1, 5, 10))
    any_moment = next(r for r in table if r["video"] == "all" and r["layer"] == "any"
                      and r["granularity"] == "moment")
    assert any_moment["r@5"] == 0.5 and any_moment["r@10"] == 1.0
    assert attribution(rows, ("semantic", "ocr")) == {"semantic": 1, "ocr": 1}


# ---------------------------------------------------------------- lexical layers

def test_tokenize_drops_stopwords_and_folds_plurals():
    assert tokenize("The Upgrades of broadside cannons") == ["upgrade", "broadside", "cannon"]
    assert tokenize("взлом реликов и даток") == ["взлом", "реликов", "даток"]


def test_bm25_ranks_the_document_sharing_rare_terms_first():
    layer = Bm25Layer([
        (m(B, 0, 5), "sailing ship sea"),
        (m(A, 10, 15), "HARBOURMASTER Armored Hull V COST Reales Upgrade"),
        (m(C, 0, 5), "ship ship ship"),
    ])
    hits = layer.search("armored hull upgrade at the harbourmaster", 5)
    assert hits[0].name == A
    assert all(h.name != B for h in hits)


# ---------------------------------------------------------------- gold + review

def test_clean_label_keeps_the_picture_and_drops_spoken_words_and_tags():
    assert clean_label("фр 02 Mackinaw и Porpoise в поясе — «пожил и в хайсеке» [личное]") == \
        "Mackinaw и Porpoise в поясе"
    assert clean_label("фр 10 логотип Минматар — «минматарский индус» (5:55,8) [личное]") == \
        "логотип Минматар"
    assert clean_label("фр19 Blue Hole — значок на карте (нырка нет в записях)") == \
        "Blue Hole — значок на карте"
    assert clean_label("фр10 «Галка» у форта") == "«Галка» у форта"


def test_items_from_events_groups_same_label_and_flags_catalog():
    events = [
        {"start_sec": 0, "end_sec": 5, "clip_start": 10, "path": "E:\\x\\" + C, "label": "фр 31 червоточина [личное]"},
        {"start_sec": 5, "end_sec": 9, "clip_start": 0, "path": "E:\\y\\Old.mp4", "label": "фр 41 червоточина [иллюстрация]"},
    ]
    items = items_from_events("wtl", events, {name_key(C)})
    assert len(items) == 1
    assert items[0]["id"] == "wtl-e001"
    assert [p["in_catalog"] for p in items[0]["picks"]] == [True, False]
    assert items[0]["picks"][0] == {"file": C, "start": 10.0, "end": 15.0, "in_catalog": True}


def test_review_round_trip_marks_wrong_edits_and_reviewed_sections():
    shots = {"_note": "x", "1": ["ship at sea"], "2": ["upgrade menu"], "3": ["fort"]}
    curation = {"1": [A, 0, 10], "2": [A, 20, 30], "3": [B, 0, 9]}
    items = items_from_shots("gm", shots, curation, {name_key(A), name_key(B)})
    gold = {"videos": [{"id": "gm", "title": "GM", "query_source": "shots.json"}],
            "items": [{"game": "assassins_creed_4", **it} for it in items]}
    sheet = render_sheet(gold)
    sheet = sheet.replace("| gm-f02 | 2 | upgrade menu | Ship Upgrade Trading 1 @ 0:20–0:30 |  |  |",
                          "| gm-f02 | 2 | upgrade menu | Ship Upgrade Trading 1 @ 0:20–0:30 |  | x |")
    sheet = sheet.replace("| gm-f03 | 3 | fort | Battle Jackdaw 1 @ 0:00–0:09 |  |  |",
                          "| gm-f03 | 3 | fort | Battle Jackdaw 1 @ 0:00–0:09 |  | fort walls under cannon fire |")
    unreviewed, counts = apply_review(gold, sheet)
    assert counts == {"ok": 0, "wrong": 1, "edited": 1, "cleared": 0, "untouched": 1}

    reviewed, counts = apply_review(gold, sheet.replace("- [ ] Раздел просмотрен", "- [x] Раздел просмотрен"))
    status = {it["id"]: it["status"] for it in reviewed["items"]}
    assert status == {"gm-f01": "ok", "gm-f02": "wrong", "gm-f03": "ok"}
    edited = next(it for it in reviewed["items"] if it["id"] == "gm-f03")
    assert edited["queries"] == ["fort walls under cannon fire"]
    assert edited["query_source"] == "shots.json+owner"
    assert gold["items"][2]["queries"] == ["fort"]  # input left untouched


def test_rebuilding_gold_keeps_owner_verdicts_and_edited_queries():
    fresh = {"items": [
        {"id": "gm-f01", "queries": ["ship"], "query_source": "shots.json", "status": "unreviewed"},
        {"id": "gm-f02", "queries": ["menu"], "query_source": "shots.json", "status": "unreviewed"},
        {"id": "gm-f03", "queries": ["fort"], "query_source": "shots.json", "status": "unreviewed"},
    ]}
    old = {"items": [
        {"id": "gm-f01", "queries": ["ship"], "query_source": "shots.json", "status": "wrong"},
        {"id": "gm-f02", "queries": ["upgrade menu, reales"], "query_source": "shots.json+owner", "status": "ok"},
    ]}
    merged = {it["id"]: it for it in merge_reviews(fresh, old)["items"]}
    assert merged["gm-f01"]["status"] == "wrong"
    assert merged["gm-f02"]["queries"] == ["upgrade menu, reales"]
    assert merged["gm-f03"]["status"] == "unreviewed"
    assert fresh["items"][0]["status"] == "unreviewed"


def test_sheet_found_column_names_each_layer():
    items = items_from_shots("gm", {"1": ["ship"]}, {"1": [A, 0, 10]}, {name_key(A)})
    gold = {"videos": [{"id": "gm", "title": "GM", "query_source": "shots.json"}], "items": items}
    results = {"items": [{"id": "gm-f01", "ranks": {"semantic": {"file": 2, "moment": None},
                                                     "ocr": {"file": None, "moment": None},
                                                     "scenes": {"file": 1, "moment": 1}}}]}
    assert "| Sf2 O· C1 |" in render_sheet(gold, results)


def test_ranked_moments_grows_until_ten_distinct_files():
    many = [m(A, t, t + 5) for t in range(0, 2000, 5)] + [m(f"v{i}.mp4", 0, 5) for i in range(12)]

    class Capped:
        def search(self, query, n):
            return many[:n]

    moments = ranked_moments(Capped(), "q", fetch=50)
    assert len({name_key(x.name) for x in moments}) >= 10
    assert file_rank(moments, Pick("v8.mp4", 0, 5)) == 10


def test_cleared_x_retracts_a_wrong_verdict_even_in_an_unticked_section():
    items = items_from_shots("gm", {"1": ["ship"]}, {"1": [A, 0, 10]}, {name_key(A)})
    gold = {"videos": [{"id": "gm", "title": "GM", "query_source": "shots.json"}],
            "items": [it | {"status": "wrong"} for it in items]}
    sheet = render_sheet(gold).replace("|  | x |", "|  |  |")
    new, counts = apply_review(gold, sheet)
    assert new["items"][0]["status"] == "unreviewed" and counts["cleared"] == 1


def test_verdict_is_not_carried_to_an_item_whose_picks_changed():
    pick = {"file": A, "start": 0.0, "end": 5.0, "in_catalog": True}
    old = {"items": [{"id": "x-e001", "queries": ["q"], "query_source": "label",
                      "picks": [pick], "status": "wrong"}]}
    moved = {"items": [{"id": "x-e001", "queries": ["other"], "query_source": "label",
                        "picks": [pick | {"start": 40.0, "end": 45.0}], "status": "unreviewed"}]}
    assert merge_reviews(moved, old)["items"][0]["status"] == "unreviewed"


def test_empty_curation_entry_is_not_a_gold_item():
    # curation.json null = fragment deliberately left without footage: no pick to grade.
    items = items_from_shots("gm", {"1": ["ship"], "2": ["fort"]}, {"1": [A, 0, 10], "2": None},
                             {name_key(A)})
    assert [it["fragment"] for it in items] == [1]
