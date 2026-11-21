"""
Shot type classifier for production plan fragments.

Classifies Russian gaming text fragments into shot types using keyword
heuristics. Also translates Russian fragments into English queries
suitable for stock footage APIs.
"""
from __future__ import annotations

import re

SHOT_TYPES = {
    "gameplay": "Gameplay footage — active game playing, fights, exploration",
    "cinematic": "Cinematic/trailer — official game cinematics, cutscenes",
    "broll_abstract": "Abstract B-roll — mood shots, transitions, atmospheric visuals",
    "broll_gaming": "Gaming B-roll — generic gaming context (controller, screen, setup)",
    "talking_head": "Talking head — narrator on camera",
    "screen_recording": "Screen recording — UI, menus, store pages, reviews",
    "comparison": "Side-by-side — before/after, version comparison",
    "text_overlay": "Text/stats overlay — numbers, dates, facts displayed on screen",
}

_GAMEPLAY_KEYWORDS = {
    "бой", "сражение", "фарм", "гринд", "прокачка", "лут", "билд", "крафт",
    "босс", "данж", "подземелье", "рейд", "арена", "пвп", "квест", "миссия",
    "убил", "убить", "зачистил", "пройти", "прошел", "прошёл", "прохождение",
    "фарминг", "респаун", "моб", "монстр", "враг", "атака", "удар", "скилл",
    "способность", "класс", "персонаж", "левел", "уровень", "снаряжение",
    "оружие", "доспехи", "щит", "меч", "лук", "магия", "заклинание",
    "хп", "мана", "стамина", "buff", "nerf", "dps", "танк", "хилер",
}

_CINEMATIC_KEYWORDS = {
    "трейлер", "кинематик", "ролик", "анонс", "катсцена", "кат-сцена",
    "заставка", "интро", "финал", "сюжет", "история", "лор", "нпс",
    "диалог", "сцена", "эпизод", "кампания",
}

_META_KEYWORDS = {
    "рейтинг", "обзор", "оценка", "metacritic", "steam", "стим", "цифры",
    "статистика", "онлайн", "продажи", "отзывы", "критики", "баллы",
    "dlc", "дополнение", "патч", "обновление", "сезон", "season",
    "цена", "стоимость", "скидка", "магазин",
}

_ATMOSPHERIC_KEYWORDS = {
    "атмосфера", "настроение", "переход", "пауза", "тишина", "эмоция",
    "музыка", "звук", "красота", "пейзаж", "мир", "вселенная",
    "эпоха", "время", "история", "ностальгия", "воспоминания",
}

_PERSONAL_KEYWORDS = {
    "я играл", "я прошел", "я прошёл", "мой опыт", "я купил", "я решил",
    "мне понравилось", "мне не понравилось", "я считаю", "по моему",
    "по-моему", "я пропустил", "я вернулся", "я начал",
}

# Russian → English stock query translation map
_STOCK_TRANSLATIONS = {
    "бой": "battle combat",
    "сражение": "epic battle",
    "фарм": "grinding loot",
    "босс": "boss fight epic",
    "подземелье": "dark dungeon",
    "магия": "magic spells fantasy",
    "тёмное фэнтези": "dark fantasy",
    "мрачный": "dark gloomy atmosphere",
    "эпичный": "epic cinematic",
    "средневековье": "medieval castle",
    "демон": "demon hell fire",
    "ангел": "angel divine light",
    "дракон": "dragon fire",
    "руины": "ancient ruins",
    "пустыня": "desert wasteland",
    "лес": "dark forest",
    "горы": "mountain landscape",
    "океан": "ocean waves",
    "космос": "space nebula stars",
    "город": "city night lights",
    "война": "war battlefield",
    "смерть": "death skull darkness",
    "победа": "victory triumph",
    "путешествие": "journey adventure",
    "ностальгия": "nostalgia retro",
    "геймер": "gaming setup gamer",
    "компьютер": "computer gaming setup",
    "контроллер": "game controller hands",
    "экран": "computer screen monitor",
}

# Game-specific terms → English for stock/YouTube
_GAME_STOCK_MAP = {
    "diablo": "diablo dark fantasy demon hell",
    "diablo4": "diablo 4 sanctuary hell demon",
    "elden_ring": "elden ring dark souls fantasy",
    "eve_online": "eve online space mmo fleet",
}


def classify_fragment(text: str, game: str | None = None) -> list[tuple[str, float]]:
    """Classify a text fragment into shot types with confidence scores.

    Returns ranked list of (shot_type, confidence) pairs.
    """
    text_lower = text.lower()
    words = set(re.findall(r"[а-яёa-z0-9]+", text_lower))

    scores: dict[str, float] = {t: 0.0 for t in SHOT_TYPES}

    gameplay_hits = words & _GAMEPLAY_KEYWORDS
    if gameplay_hits:
        scores["gameplay"] = min(len(gameplay_hits) * 0.2, 1.0)

    cinematic_hits = words & _CINEMATIC_KEYWORDS
    if cinematic_hits:
        scores["cinematic"] = min(len(cinematic_hits) * 0.25, 1.0)

    meta_hits = words & _META_KEYWORDS
    if meta_hits:
        scores["screen_recording"] = min(len(meta_hits) * 0.25, 0.9)
        scores["text_overlay"] = min(len(meta_hits) * 0.15, 0.7)

    atmo_hits = words & _ATMOSPHERIC_KEYWORDS
    if atmo_hits:
        scores["broll_abstract"] = min(len(atmo_hits) * 0.2, 0.9)

    for phrase in _PERSONAL_KEYWORDS:
        if phrase in text_lower:
            scores["talking_head"] = max(scores["talking_head"], 0.6)
            break

    if re.search(r"(сравни|до и после|раньше.*теперь|было.*стало)", text_lower):
        scores["comparison"] = 0.7

    if game and scores["gameplay"] < 0.3 and not any(
        s > 0.5 for k, s in scores.items() if k != "gameplay"
    ):
        scores["gameplay"] = 0.4

    if max(scores.values()) < 0.2:
        scores["broll_abstract"] = 0.3

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(t, s) for t, s in ranked if s > 0.0]


def extract_stock_query(text: str, game: str | None = None) -> str:
    """Translate a Russian text fragment into an English stock search query.

    Prioritizes generic visual concepts over specific game names (stock
    footage won't have game-specific content anyway).
    """
    text_lower = text.lower()
    query_parts: list[str] = []

    for ru_word, en_terms in _STOCK_TRANSLATIONS.items():
        if ru_word in text_lower:
            query_parts.append(en_terms)

    if not query_parts and game:
        query_parts.append(_GAME_STOCK_MAP.get(game, "gaming"))

    if not query_parts:
        query_parts.append("dark atmosphere cinematic")

    en_words = re.findall(r"[A-Z][a-zA-Z]{2,}", text)
    if en_words:
        query_parts.append(" ".join(en_words[:3]))

    combined = " ".join(query_parts)
    words = combined.split()
    seen: set[str] = set()
    unique: list[str] = []
    for w in words:
        wl = w.lower()
        if wl not in seen:
            seen.add(wl)
            unique.append(w)
    return " ".join(unique[:8])


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Shot type and stock query for narration fragments "
                                 "(no arguments: a built-in Russian demo)")
    ap.add_argument("fragment", nargs="*", help="narration fragments to classify")
    ap.add_argument("--game", default="diablo4", help="game key for the stock query hint")
    args = ap.parse_args()
    test_fragments = args.fragment or [
        "Я вернулся в Diablo IV после полутора лет перерыва, когда вышел Lord of Hatred.",
        "Босс был настолько сложный, что я потратил три часа на одну попытку прохождения.",
        "По оценкам Steam и Metacritic, дополнение получило 85 баллов.",
        "Кинематик вступительной сцены просто потрясающий, такой атмосферный ролик.",
    ]
    for frag in test_fragments:
        print(f"Text: {frag[:60]}...")
        types = classify_fragment(frag, game=args.game)
        print(f"  Types: {[(t, f'{s:.2f}') for t, s in types[:3]]}")
        print(f"  Stock query: {extract_stock_query(frag, game=args.game)}")
        print()
