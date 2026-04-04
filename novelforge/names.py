"""Genre-aware character name pool for NovelForge.

Provides pre-generated pools of Anglo-Saxon–inspired first and last names,
organised by genre group and gender.  Names are chosen to be distinctive but
pronounceable, and are stylistically calibrated per genre:

* ``scifi``       – futuristic / lightly alien-sounding (modified Anglo-Saxon roots)
* ``fantasy``     – archaic Old English / Norse-inspired
* ``historical``  – authentic Anglo-Saxon / medieval English
* ``gothic``      – atmospheric Victorian / archaic dark-toned
* ``contemporary``– modern but uncommon Anglo-Saxon-derived
* ``thriller``    – strong, grounded, hard-edged Anglo-Saxon
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Genre → style-group mapping
# ---------------------------------------------------------------------------
_GENRE_GROUP: dict[str, str] = {
    # Futuristic / speculative
    "Science Fiction":    "scifi",
    "Speculative Fiction": "scifi",
    "Dystopian":          "scifi",
    # Fantastical
    "Fantasy":            "fantasy",
    "Urban Fantasy":      "fantasy",
    "Paranormal":         "fantasy",
    "Magical Realism":    "fantasy",
    # Historical / period
    "Historical Fiction": "historical",
    "Western":            "historical",
    # Atmospheric / gothic
    "Gothic Fiction":     "gothic",
    # Contemporary / literary
    "Contemporary Fiction": "contemporary",
    "Literary Fiction":     "contemporary",
    "Romance":              "contemporary",
    "Young Adult":          "contemporary",
    "Satire Humor":         "contemporary",
    # Suspense / dark
    "Crime":     "thriller",
    "Mystery":   "thriller",
    "Thriller":  "thriller",
    "Noir":      "thriller",
    "Horror":    "thriller",
    "Adventure": "thriller",
}

# ---------------------------------------------------------------------------
# Name pool data
# Structure: _NAMES[group][key] = list[str]
# Keys: "male_first", "female_first", "last"
# ---------------------------------------------------------------------------
_NAMES: dict[str, dict[str, list[str]]] = {
    # -------------------------------------------------------------------
    # Science Fiction – futuristic / lightly alien-sounding names built
    # from Anglo-Saxon roots: short syllables, unusual endings, modified
    # consonant clusters.
    # -------------------------------------------------------------------
    "scifi": {
        "male_first": [
            "Aldren", "Brant", "Calix", "Cael", "Dax", "Drevan",
            "Jex", "Kylen", "Lexon", "Orrin", "Oryn", "Ryn",
            "Talos", "Theron", "Vance", "Vex", "Voren", "Xander",
            "Zel", "Cynric",
        ],
        "female_first": [
            "Aeryn", "Avara", "Bryn", "Cala", "Ciren", "Cyla",
            "Dara", "Luen", "Orla", "Rynn", "Sylvae", "Taya",
            "Vael", "Vesna", "Vesper", "Yael", "Zira", "Cyra",
            "Tayne", "Endra",
        ],
        "last": [
            "Ashford", "Brand", "Cole", "Croft", "Cross",
            "Drake", "Draven", "Fell", "Holt", "Kyne",
            "Roe", "Ryne", "Stone", "Stroud", "Thane",
            "Vane", "Voss", "Weld", "Wyn", "Yeld",
        ],
    },

    # -------------------------------------------------------------------
    # Fantasy – archaic Old English, Norse, and Arthurian-inspired names.
    # -------------------------------------------------------------------
    "fantasy": {
        "male_first": [
            "Aldric", "Branoc", "Caedmon", "Cenwulf", "Edwyn",
            "Eogan", "Godwin", "Halvard", "Leofric", "Osric",
            "Ranulf", "Sigebert", "Styrr", "Thane", "Ulfric",
            "Wystan", "Eadric", "Gareth", "Hadden", "Odran",
        ],
        "female_first": [
            "Aelith", "Brenna", "Caoimhe", "Ceridwen", "Elspeth",
            "Fendrel", "Gwyneth", "Hild", "Isolde", "Morwenna",
            "Niamh", "Rowena", "Sigrid", "Sunniva", "Wulfwyn",
            "Branwen", "Deirdre", "Etaine", "Theda", "Freya",
        ],
        "last": [
            "Ashveil", "Blackthorn", "Dawnmere", "Dunhollow",
            "Embervale", "Forrest", "Goldenbrook", "Greymoor",
            "Hawthorn", "Ironfell", "Longmire", "Mosshollow",
            "Nightshade", "Oakhurst", "Ravenwood", "Silverfen",
            "Stonemark", "Thicket", "Windsorrow", "Wyldmere",
        ],
    },

    # -------------------------------------------------------------------
    # Historical – authentic Anglo-Saxon / medieval English names.
    # -------------------------------------------------------------------
    "historical": {
        "male_first": [
            "Aelfric", "Baldric", "Cerdic", "Dunstan", "Edmund",
            "Egbert", "Godric", "Harold", "Leofstan", "Oswald",
            "Raedwald", "Sigeweard", "Swithun", "Wulfhere", "Wulfnoth",
            "Aethelred", "Brictmer", "Ceolwulf", "Eadmund", "Hereward",
        ],
        "female_first": [
            "Aelfflaed", "Aldgyth", "Cyneswith", "Eadgyth", "Edith",
            "Elfgifu", "Godgifu", "Gunnhild", "Leofgifu", "Mildthryth",
            "Osgyth", "Sigeflead", "Wulfgifu", "Aethelflaed", "Burghild",
            "Cwenthryth", "Ealdswith", "Frithswith", "Tova", "Aelswith",
        ],
        "last": [
            "Athelworth", "Brockwell", "Caldwell", "Dunmore",
            "Eastham", "Foxley", "Greystone", "Halewood",
            "Ironbridge", "Ketterwell", "Longford", "Norwood",
            "Pennworth", "Ravenscroft", "Stanwick", "Thornbury",
            "Underhill", "Wentworth", "Yarwell", "Zetford",
        ],
    },

    # -------------------------------------------------------------------
    # Gothic – atmospheric, dark-toned Anglo-Saxon / Victorian names.
    # -------------------------------------------------------------------
    "gothic": {
        "male_first": [
            "Aldous", "Barnabas", "Corvus", "Dorian", "Everard",
            "Fabian", "Gideon", "Hadley", "Isidore", "Jasper",
            "Lement", "Mordecai", "Oberon", "Phineas", "Quentin",
            "Remiel", "Silas", "Thaddeus", "Ulric", "Alistair",
        ],
        "female_first": [
            "Arabella", "Beatrix", "Cordelia", "Desdemona", "Evangeline",
            "Felicity", "Genevieve", "Honoria", "Isadora", "Lavinia",
            "Millicent", "Nerissa", "Octavia", "Patience", "Rowena",
            "Sophronia", "Thessaly", "Ursula", "Vivienne", "Winifred",
        ],
        "last": [
            "Ashmore", "Blackwell", "Cravenhollow", "Darkwater",
            "Evenmoore", "Foxworth", "Greymantle", "Havenshade",
            "Irongate", "Kellsworth", "Lockwood", "Morrowfen",
            "Nightfall", "Osbourne", "Pallsworth", "Ravenscroft",
            "Shadwick", "Thornwood", "Umberton", "Veilbrook",
        ],
    },

    # -------------------------------------------------------------------
    # Contemporary – modern but uncommon Anglo-Saxon-derived names.
    # -------------------------------------------------------------------
    "contemporary": {
        "male_first": [
            "Aldric", "Beckett", "Cormac", "Declan", "Emmet",
            "Fletcher", "Gareth", "Hadley", "Idris", "Jasper",
            "Keane", "Lachlan", "Merritt", "Nash", "Orson",
            "Phelan", "Quinn", "Rafferty", "Sullivan", "Theron",
        ],
        "female_first": [
            "Blythe", "Celia", "Delia", "Edith", "Fern",
            "Greta", "Harriet", "Ingrid", "Juno", "Liesel",
            "Maren", "Nell", "Opal", "Petra", "Ros",
            "Sable", "Thea", "Verity", "Wren", "Zephyrine",
        ],
        "last": [
            "Alderton", "Barlow", "Corwin", "Davenport", "Eastwood",
            "Fairfax", "Granger", "Hartley", "Inwood", "Kelsey",
            "Langford", "Mercer", "Norford", "Ogden", "Pemberton",
            "Radford", "Sutton", "Tilbury", "Underwood", "Whitmore",
        ],
    },

    # -------------------------------------------------------------------
    # Thriller – strong, grounded Anglo-Saxon names with gravitas.
    # Short, hard-edged, memorable.
    # -------------------------------------------------------------------
    "thriller": {
        "male_first": [
            "Alden", "Brett", "Caden", "Dane", "Emerson",
            "Ford", "Grant", "Harlan", "Ivor", "Jensen",
            "Kent", "Lance", "Mack", "Noel", "Owen",
            "Pierce", "Reid", "Sawyer", "Trent", "Ward",
        ],
        "female_first": [
            "Adler", "Blair", "Carson", "Drew", "Eden",
            "Fallon", "Glen", "Harlow", "Iris", "Jem",
            "Kerr", "Leigh", "Maren", "Nora", "Peyton",
            "Quinn", "Rayne", "Scout", "Tess", "Vance",
        ],
        "last": [
            "Arden", "Blake", "Chase", "Dent", "Egan",
            "Fenn", "Gale", "Hale", "Ingram", "Jove",
            "Keel", "Lorne", "Marsh", "Nance", "Orryn",
            "Page", "Rook", "Strode", "Thurn", "Vale",
        ],
    },
}

# Human-readable style descriptions per group, used in prompt text.
_STYLE_DESC: dict[str, str] = {
    "scifi": (
        "futuristic or lightly alien-sounding names built from Anglo-Saxon roots "
        "(e.g. modified consonant clusters, short syllables, unusual endings)"
    ),
    "fantasy": (
        "archaic Anglo-Saxon, Old English, or Norse-inspired names "
        "(e.g. Aldric, Ceridwen, Ranulf)"
    ),
    "historical": (
        "authentic Anglo-Saxon or medieval English names that suit the period "
        "(e.g. Aelfric, Eadgyth, Dunstan)"
    ),
    "gothic": (
        "atmospheric Victorian or archaic names with dark resonance "
        "(e.g. Dorian, Evangeline, Thaddeus)"
    ),
    "contemporary": (
        "modern but uncommon Anglo-Saxon-derived names — distinctive without "
        "sounding invented (e.g. Beckett, Verity, Sullivan)"
    ),
    "thriller": (
        "strong, grounded Anglo-Saxon names with gravitas — short, hard-edged, "
        "memorable (e.g. Harlan, Tess, Reid)"
    ),
}


def _group_for_genre(genre: str) -> str:
    """Return the style-bucket name for a given genre string."""
    return _GENRE_GROUP.get(genre, "contemporary")


def get_name_pool(genre: str) -> dict[str, list[str]]:
    """Return the name pool appropriate for *genre*.

    Returns a dict with keys ``"male_first"``, ``"female_first"``, ``"last"``,
    each mapping to a list of name strings.
    """
    group = _group_for_genre(genre)
    source = _NAMES[group]
    return {key: list(names) for key, names in source.items()}


def format_name_pool_for_prompt(genre: str) -> str:
    """Return a human-readable string suitable for embedding in an LLM prompt.

    The text describes the naming style expected for *genre* and lists the
    pre-generated pool that the LLM should draw from (or use as stylistic
    inspiration when inventing new names).
    """
    group = _group_for_genre(genre)
    pool = _NAMES[group]
    desc = _STYLE_DESC.get(group, _STYLE_DESC["contemporary"])

    male_names = ", ".join(pool["male_first"])
    female_names = ", ".join(pool["female_first"])
    last_names = ", ".join(pool["last"])

    return (
        f"Use {desc}.\n"
        f"Pre-generated name pool — draw from these or invent similar names in the same style:\n"
        f"  Male first names:   {male_names}\n"
        f"  Female first names: {female_names}\n"
        f"  Last names:         {last_names}"
    )
