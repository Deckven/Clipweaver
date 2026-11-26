"""
Game-specific vocabularies fed to the VLM as context hints.

Keep each list focused (~50-80 most common items). Bigger lists eat context
without helping much — the goal is to nudge the model toward correct names,
not to enumerate every entity.
"""

import re
from difflib import SequenceMatcher

ELDEN_RING = {
    "description": (
        "Elden Ring is a dark-fantasy action RPG. Typical visual scenes: a knight "
        "(Tarnished) exploring an open world, boss fights, melee combat with "
        "swords and spells, horse riding on Torrent, NPCs at sites of grace, "
        "castle ruins, forests, swamps, caves. UI is minimal."
    ),
    "regions": [
        "Limgrave", "Weeping Peninsula", "Stormveil Castle", "Caelid",
        "Dragonbarrow", "Redmane Castle", "Sellia", "Liurnia",
        "Raya Lucaria", "Caria Manor", "Altus Plateau", "Leyndell",
        "Capital Outskirts", "Mt. Gelmir", "Volcano Manor",
        "Mountaintops of the Giants", "Castle Sol", "Consecrated Snowfield",
        "Miquella's Haligtree", "Elphael", "Crumbling Farum Azula",
        "Nokron", "Nokstella", "Siofra River", "Ainsel River",
        "Deeproot Depths", "Castle Morne", "Stormhill", "Dragon Hill",
        "Land of Shadow", "Scadu Altus", "Belurat",
    ],
    "bosses": [
        "Margit", "Godrick", "Rennala", "Radahn", "Starscourge Radahn",
        "Morgott", "Mohg", "Malenia", "Maliketh", "Godfrey",
        "Hoarah Loux", "Radagon", "Elden Beast", "Astel", "Fire Giant",
        "Rykard", "Placidusax", "Loretta", "Commander Niall",
        "Commander O'Neil", "Crucible Knight", "Tree Sentinel",
        "Red Wolf of Radagon", "Royal Knight Loretta", "Godskin Apostle",
        "Godskin Noble", "Godskin Duo", "Black Blade Kindred",
        "Lichdragon Fortissax", "Dragonlord Placidusax", "Messmer",
        "Bayle", "Promised Consort Radahn", "Rellana",
    ],
    "enemies": [
        "Tarnished", "Misbegotten", "Crystallian", "Erdtree Avatar",
        "Dragonkin Soldier", "Tibia Mariner", "Lion Guardian",
        "Death Bird", "Death Rite Bird", "Tree Spirit",
        "Ulcerated Tree Spirit", "Runebear", "Bloodhound Knight",
        "Night's Cavalry", "Magma Wyrm", "Ancestor Spirit",
        "Putrid Avatar", "Scarlet Rot lake",
    ],
    "concepts": [
        "scarlet rot", "Golden Order", "Erdtree", "rune arc",
        "site of grace", "stake of Marika", "spirit ashes",
        "ashes of war", "talisman", "Flask of Wondrous Physick",
        "Torrent (horse)", "summoning pool", "great rune",
        "Frenzied Flame", "Three Fingers", "Two Fingers",
    ],
    "actions": [
        "boss_fight", "exploration", "horseback_riding", "stealth",
        "parry", "backstab", "spell_casting", "incantation",
        "dialogue", "cutscene", "death", "level_up", "item_pickup",
        "summoning", "open_world_traversal", "dungeon_crawl",
    ],
}


EVE_ONLINE = {
    "description": (
        "Eve Online is a science-fiction space MMO (NOT a fantasy game). "
        "Typical visual scenes: spaceships flying in space, starfields with stars "
        "and nebulae, space stations and citadels (Fortizar, Astrahus), station "
        "interiors (hangar view with the player's ship), HUD overlays with windows "
        "(market, inventory, fitting, overview, chat, star map, probe scanner), "
        "wormhole entries, abyssal deadspace pockets with floating obstacles, "
        "mining lasers on asteroids, warp tunnels (blue streaks), jump gate "
        "transitions. There are NO swords, melee combat, magic spells, characters "
        "on foot, or fantasy creatures. What looks like a 'glowing device' is "
        "almost certainly a UI element or a ship module effect. What looks like "
        "an 'enemy' is another ship or NPC pirate vessel."
    ),
    "regions": [
        "Jita", "Amarr", "Dodixie", "Rens", "Hek",
        "Thera", "Turnur",
        "Highsec", "Lowsec", "Nullsec", "Wormhole space", "Pochven",
        "New Eden", "Jove Empire", "Abyssal Deadspace",
        # Systems on the Gallente Election Campaign Trail route
        "Luminaire", "Villore", "Caldari Prime", "Gallente Prime",
        "Old Man Star", "Oueberia",
        "Postouvin", "Gererique", "Molea", "Gidali",
        "Kor-Azor Prime", "Sarum Prime", "Mekhios", "Thebeka",
        "Nafomeh", "Saminer", "Turba", "Mozzidit",
        "Dead End", "Carirgnottin", "Carignotin",
        # Constellations / regions mentioned in lore
        "Essence", "Domain", "Genesis", "Tash-Murkon", "Kador",
        "Hanid", "Solitude", "Intaki Syndicate", "Thera Space",
    ],
    "factions": [
        "Gallente", "Caldari", "Amarr", "Minmatar",
        "Jove", "Triglavian", "EDENCOM",
        "Guristas", "Serpentis", "Angel Cartel", "Blood Raiders",
        "Sansha's Nation", "Mordu's Legion", "CONCORD",
        "Drifters", "Sleepers", "Rogue Drones",
    ],
    "events": [
        "Gallente President Election", "Caldari Prime", "Federation Day",
        "Crimson Harvest", "Yoiul Festival", "Scope Network",
        "AIR Carriers", "AIR system Rewards",
        "Pochven invasion", "Triglavian invasion",
    ],
    "ships": [
        "Astero", "Stratios", "Nestor",
        "Buzzard", "Heron", "Magnate", "Probe",
        "Hawk", "Kestrel", "Merlin", "Cormorant", "Caracal", "Drake", "Raven", "Golem",
        "Vexor", "Imicus", "Tristan", "Algos", "Catalyst",
        "Ishtar", "Myrmidon", "Dominix", "Hyperion", "Megathron",
        "Proteus", "Legion", "Loki", "Tengu",
        "Cerberus", "Eagle", "Sacrilege",
        "Retriever", "Covetor", "Procurer", "Skiff",
        "Hulk", "Mackinaw", "Mackinaw",
        "Porpoise", "Orca", "Rorqual",
        "Venture", "Prospect", "Endurance",
        "Praxis", "Sunesis", "Gnosis",
        "Tayra", "Epithal", "Occator", "Badger", "Iteron",
        "Deluge", "Torrent", "Squall",
        "Ragnarok", "Leviathan", "Avatar", "Erebus",
        "Vargur", "Paladin", "Kronos", "Golem",
        "Onyx", "Broadsword", "Devoter", "Phobos",
        "Fortizar", "Astrahus", "Azbel", "Raitaru", "Sotiyo", "Keepstar",
        "Metenox", "Metox",
        "Vanya", "Pod", "Capsule",
    ],
    "mechanics": [
        "Abyssal Deadspace", "Abyss", "T1 Abyss", "T2 Abyss", "T3 Abyss", "T4 Abyss", "T5 Abyss",
        "wormhole", "WH transit", "WH rolling", "collapsing WH",
        "mining", "ice mining", "moon mining", "ore mining",
        "manufacturing", "reprocessing", "planetary interaction", "PI",
        "ratting", "relic site", "data site", "anomaly",
        "PvP", "PvE", "fleet", "gate camp",
        "eviction", "citadel defense", "structure bash",
        "jump clone", "capsule", "implants",
        "exploration", "hacking", "scanning",
        "mission", "level 4 mission", "agent", "storyline",
        "market", "trading", "contracts",
        "skill queue", "skill extractor", "skill injector",
        "docking", "undocking", "warp", "jump gate",
    ],
    "ores_minerals": [
        "Veldspar", "Scordite", "Pyroxeres", "Plagioclase",
        "Omber", "Kernite", "Jaspet", "Hemorphite", "Hedbergite",
        "Gneiss", "Dark Ochre", "Spodumain", "Crokite", "Bistot", "Arkonor", "Mercoxit",
        "Moon ore", "Asteroid belt",
    ],
    "landmarks": [
        # Campaign Trail beacons and monuments
        "Campaign Trail", "Starting Line", "Finish Line",
        "Chribba Monument", "Chribba", "Veldnaught",
        "Doriam II Memorial", "Doriam II", "Doriam",
        "Dorean II", "Dorean",
        "Black Monolith", "Monolith",
        "Fallen Capsuleer Memorial", "Capsuleer Memorial",
        "H4RP-4 Kyonoke Research Memorial", "H4RP-4", "Kyonoke",
        "Duvolle Observatory", "Duvolle Laboratories", "Duvolle",
        "Traumark Installation", "Traumark",
        "Old Man Star Gate", "Darieu Gate", "Soere Darieux", "Darieux",
        "EVE Gate", "New Eden Gate", "Gates of New Eden",
        "Mekhios Graveyard", "Battle of Mekhios",
        "Tetrimon", "Order of Saint Tetrimon",
        "EVE Travel Agency", "EVE Travel",
        "Talocan", "Speaker of Truths",
        "Jamyl Sarum", "Jamyl I",
        "Abaddon", "Apocalypse",
        "Alixandra Moreau", "Jacus Roden", "Celes Aguard",
    ],
}


EVE_FRONTIER = {
    "description": (
        "EVE Frontier is a sci-fi survival sandbox game set in a new star cluster "
        "disconnected from New Eden. Players build, craft, gather resources, and "
        "establish settlements on planets and in space. It uses blockchain-based "
        "smart assemblies for player-driven economies. Typical scenes: planetary "
        "surfaces with resource gathering, base building, space flight, crafting "
        "interfaces, smart assembly programming, survival mechanics. Visually "
        "distinct from EVE Online — more grounded, survival-focused, with on-foot "
        "gameplay and planetary exploration alongside spaceflight."
    ),
    "mechanics": [
        "Smart Assembly", "crafting", "resource gathering", "base building",
        "planetary exploration", "survival", "spaceflight", "settlement",
        "economy", "trading", "mining", "exploration",
    ],
    "events": [
        "EVE Fanfest 2026", "Keynote", "CCP Games",
    ],
}


EVE_VANGUARD = {
    "description": (
        "EVE Vanguard is a first-person shooter extraction game set in the EVE "
        "Online universe. Players are Warclone soldiers deployed to planetary "
        "surfaces to complete contracts: salvage, sabotage, and combat missions. "
        "Typical scenes: FPS combat on planetary surfaces, squad-based tactical "
        "gameplay, extraction points, loot and gear management, deployment from "
        "orbit, integration with EVE Online's wars and sovereignty. Visually: "
        "sci-fi military FPS with EVE universe aesthetics — Caldari/Amarr/Gallente/ "
        "Minmatar architectural styles on ground installations."
    ),
    "mechanics": [
        "extraction", "deployment", "contracts", "salvage", "sabotage",
        "combat", "Warclone", "squad", "loot", "gear",
        "planetary surface", "orbital deployment",
    ],
    "factions": [
        "Gallente", "Caldari", "Amarr", "Minmatar",
    ],
    "events": [
        "EVE Fanfest 2026", "Keynote", "CCP Games",
    ],
}


DIABLO4 = {
    "description": (
        "Diablo IV is a dark-fantasy action RPG (isometric top-down view). "
        "Typical scenes: a character fighting hordes of demons, exploring "
        "dungeons and open world zones, boss fights with large health bars, "
        "loot drops, skill effects (fire, ice, shadow, blood, bone, curses). "
        "UI shows health/mana globes, skill bar at bottom, minimap, quest tracker. "
        "The Warlock class uses hellfire, curses, and demonic summons."
    ),
    "regions": [
        "Sanctuary", "Fractured Peaks", "Scosglen", "Dry Steppes",
        "Kehjistan", "Hawezar", "Nahantu",
        "Kyovashad", "Cerrigar", "Ked Bardu", "Gea Kul", "Zarbinzet",
        "Kurast", "Travincal", "Teganze", "Torajan Jungles",
        "Caldeum", "Ruins of Rakhat Keep",
        "Realm of Hatred", "Burning Hells", "Cathedral of Hatred",
    ],
    "bosses": [
        "Lilith", "Mephisto", "Elias", "Andariel", "Duriel",
        "Ashava", "Wandering Death", "Avarice",
        "The Butcher", "Echo of Mephisto", "Echo of Andariel",
        "Echo of Duriel", "Beast in the Ice",
        "Grigoire", "Lord Zir", "Uber Lilith",
        "Varshan", "Dark Master",
    ],
    "enemies": [
        "Fallen", "Goatmen", "Skeleton", "Zombie", "Vampire",
        "Spider", "Succubus", "Balrog", "Oppressor",
        "Blood Bishop", "Treasure Goblin", "Elite",
        "Champion", "Unique monster", "World Boss",
        "Demon", "Undead", "Drowned", "Cannibal",
        "Herald of Mephisto", "Constructs",
    ],
    "classes": [
        "Warlock", "Barbarian", "Sorcerer", "Sorceress",
        "Rogue", "Druid", "Necromancer", "Spiritborn",
    ],
    "mechanics": [
        "Helltide", "Nightmare Dungeon", "Pit of Artificer",
        "Infernal Hordes", "The Dark Citadel", "Undercity",
        "World Tier", "Torment", "Paragon Board", "Glyphs",
        "Aspects", "Codex of Power", "Tempering", "Masterworking",
        "Greater Affix", "Mythic Unique", "Runewords",
        "Mercenaries", "Kurast Undercity",
        "Skill tree", "Ultimate skill", "Passive skills",
        "Enchantment", "Legendary", "Unique item",
        "Elixir", "Potion", "Town Portal", "Waypoint",
    ],
    "actions": [
        "boss_fight", "dungeon_clear", "exploration", "helltide",
        "nightmare_dungeon", "loot", "crafting", "dialogue",
        "cutscene", "death", "level_up", "world_boss",
        "pvp", "quest", "side_quest", "campaign",
    ],
}


DIABLO = {
    "description": (
        "Diablo franchise — dark-fantasy action RPGs (isometric top-down view). "
        "Covers Diablo I (gothic horror, single cathedral dungeon), Diablo II "
        "(five acts across desert/jungle/hell), Diablo III (more colorful, "
        "large-scale demon battles). Typical scenes: hack-and-slash combat, "
        "loot drops, skill effects, town hubs, cinematics with angels and demons."
    ),
    "bosses": [
        "Diablo", "Mephisto", "Baal", "Andariel", "Duriel",
        "Belial", "Azmodan", "Malthael", "The Butcher",
        "King Leoric", "Skeleton King", "Rakanoth", "Izual",
        "Maghda", "Zoltun Kulle", "Adria", "Urzael",
    ],
    "regions": [
        "Sanctuary", "Tristram", "Kurast", "Lut Gholein",
        "Harrogath", "Arreat Summit", "Pandemonium Fortress",
        "Burning Hells", "High Heavens", "Caldeum", "Bastion's Keep",
        "Westmarch", "New Tristram", "Act I", "Act II", "Act III",
        "Act IV", "Act V",
    ],
    "classes": [
        "Warrior", "Rogue", "Sorcerer", "Amazon", "Necromancer",
        "Barbarian", "Sorceress", "Paladin", "Druid", "Assassin",
        "Witch Doctor", "Monk", "Demon Hunter", "Crusader",
    ],
}


def vocab_for(game: str) -> dict | None:
    """The entity vocabulary of a game key, or None for a game without one."""
    return {
        "elden_ring": ELDEN_RING,
        "eve_online": EVE_ONLINE,
        "eve_frontier": EVE_FRONTIER,
        "eve_vanguard": EVE_VANGUARD,
        "diablo4": DIABLO4,
        "diablo": DIABLO,
    }.get(game)


def render_vocab_hint(vocab: dict) -> str:
    """Format vocab as a compact hint for the VLM prompt."""
    lines = []
    for cat, items in vocab.items():
        if not isinstance(items, list):
            continue
        lines.append(f"  {cat}: {', '.join(items)}")
    return "\n".join(lines)


# ---------------- Path-based entity matching ----------------


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def match_entities(text: str, items: list[str], min_ratio: float = 0.85) -> list[str]:
    """Find items from a vocab list mentioned in free text.

    Two strategies combined:
      1. Substring match of normalized item inside normalized text.
      2. Word-level fuzzy match (catches typos like 'Radaganh' -> 'Radahn').
    """
    norm_text = _norm(text)
    words = re.findall(r"[A-Za-z0-9']+", text)
    matches: list[str] = []
    seen: set[str] = set()

    # Check longer items first so "Redmane Castle" wins over "castle"
    for item in sorted(items, key=len, reverse=True):
        key = item.lower()
        if key in seen:
            continue
        norm_item = _norm(item)
        if not norm_item:
            continue

        # Strategy 1: normalized substring
        if len(norm_item) >= 4 and norm_item in norm_text:
            matches.append(item)
            seen.add(key)
            continue

        # Strategy 2: fuzzy on individual words (for single-word entities)
        if " " not in item.strip():
            for w in words:
                if len(w) < 4:
                    continue
                if SequenceMatcher(None, _norm(w), norm_item).ratio() >= min_ratio:
                    matches.append(item)
                    seen.add(key)
                    break

    return matches


def match_region(folder_name: str, vocab: dict) -> str | None:
    """Return the canonical region name if the folder maps to one."""
    regions = vocab.get("regions", [])
    norm_folder = _norm(folder_name)
    if not norm_folder:
        return None
    # Exact or substring-aware match
    for r in sorted(regions, key=len, reverse=True):
        nr = _norm(r)
        if nr and (nr == norm_folder or nr in norm_folder or norm_folder in nr):
            return r
    # Fuzzy fallback
    best = None
    best_ratio = 0.0
    for r in regions:
        ratio = SequenceMatcher(None, norm_folder, _norm(r)).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best = r
    return best if best_ratio >= 0.75 else None
