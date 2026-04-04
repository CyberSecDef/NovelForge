"""
Voice seed system for NovelForge.

Selects a random tonal palette at novel creation time to ensure each novel
has a distinct prose voice, preventing the uniform "measured, steady,
deliberate" style that LLMs default to.
"""

import re
import random

# Each voice seed defines a prose style, emotional register, and
# sensory preference. One is selected per novel and injected into
# every chapter draft and refinement prompt.
_VOICE_SEEDS: list[dict[str, str]] = [
    {
        "name": "sharp_angular",
        "prose_style": (
            "Write with sharp, angular prose. Favor short declarative sentences "
            "and hard consonants. Cut adjectives ruthlessly. Let action carry "
            "emotion — show through movement, not through internal monologue. "
            "Paragraphs should hit like punches."
        ),
        "emotional_register": (
            "Characters express emotion through action, not reflection. Anger "
            "comes out as thrown objects or clipped speech. Grief shows in what "
            "characters refuse to say. Tenderness is awkward and reluctant."
        ),
        "sensory_preference": (
            "Favor sound and texture over smell. Describe the scrape of metal, "
            "the hum of machinery, the grit of sand under boots. Silence is "
            "a presence, not an absence."
        ),
    },
    {
        "name": "lyrical_flowing",
        "prose_style": (
            "Write with flowing, lyrical prose. Favor long sentences with "
            "subordinate clauses that unspool like ribbon. Use soft vowels "
            "and liquid consonants. Let descriptions breathe and expand. "
            "Occasional sentence fragments for contrast."
        ),
        "emotional_register": (
            "Characters are emotionally transparent — they feel deeply and "
            "openly. Inner monologue is rich and associative, connecting "
            "present moments to memory. Joy is effervescent. Sorrow pools "
            "and spreads slowly."
        ),
        "sensory_preference": (
            "Favor light and color. Describe the quality of afternoon light, "
            "the gradient of a sunset, the way shadows define a face. "
            "Weather mirrors emotional state without being heavy-handed."
        ),
    },
    {
        "name": "dry_wit",
        "prose_style": (
            "Write with dry wit and understatement. Favor irony and deadpan "
            "observation. The narrator notices absurdity without commenting on "
            "it directly. Humor lives in the gap between what characters say "
            "and what they mean. Keep prose lean but not stark."
        ),
        "emotional_register": (
            "Characters deflect emotion with humor and sarcasm. Vulnerability "
            "shows through failed attempts at deflection. The funniest moments "
            "are the saddest. Characters who never cry might laugh at the wrong time."
        ),
        "sensory_preference": (
            "Favor the mundane and specific — the brand of cigarette, the exact "
            "shade of institutional paint, the particular hum of a fluorescent "
            "light. Grand vistas are described with deliberate anti-climax."
        ),
    },
    {
        "name": "visceral_kinetic",
        "prose_style": (
            "Write with visceral, kinetic energy. Favor active verbs and present "
            "participles. Sentences should move — characters don't stand, they "
            "shift weight; they don't look, they track. Prose has physical "
            "momentum. Use sentence fragments during action sequences."
        ),
        "emotional_register": (
            "Characters experience emotion as physical sensation — adrenaline, "
            "nausea, the ache of tired muscles. Fear is sweat and rapid "
            "breathing, not philosophical reflection. Love is proximity "
            "and heat, not metaphor."
        ),
        "sensory_preference": (
            "Favor touch and proprioception — the weight of a pack, the pull "
            "of gravity on a turn, the cold of metal through gloves. "
            "Describe what the body feels, not what the eyes see."
        ),
    },
    {
        "name": "sparse_cinematic",
        "prose_style": (
            "Write as if directing a camera. Favor visual description and "
            "external action. Minimize internal monologue — let dialogue and "
            "behavior reveal character. Use white space. Short paragraphs. "
            "Scene breaks instead of transitions."
        ),
        "emotional_register": (
            "Characters are opaque — the reader infers emotion from gesture, "
            "silence, and what characters choose not to do. Restraint is the "
            "dominant mode. The most powerful moments are the quietest."
        ),
        "sensory_preference": (
            "Favor visual composition — framing, distance, angle. Describe "
            "scenes as a cinematographer would shoot them. Close-ups for "
            "tension, wide shots for isolation, tracking shots for pursuit."
        ),
    },
    {
        "name": "dense_literary",
        "prose_style": (
            "Write with dense, layered prose. Favor complex sentences with "
            "multiple embedded observations. Use precise vocabulary — the exact "
            "word, not the approximate one. Allusion and subtext over exposition. "
            "Each paragraph rewards rereading."
        ),
        "emotional_register": (
            "Characters intellectualize emotion but are betrayed by their "
            "bodies and habits. A character who claims to be fine will "
            "compulsively straighten objects or pick at their cuticles. "
            "Insight arrives obliquely, through metaphor and association."
        ),
        "sensory_preference": (
            "Favor taste and smell as memory triggers. Describe the terroir "
            "of a place — its particular combination of air quality, cuisine, "
            "and history. Spaces have personality and opinion."
        ),
    },
    {
        "name": "conversational_intimate",
        "prose_style": (
            "Write as if telling the story to a friend late at night. Favor "
            "a conversational, confiding tone. Use contractions, colloquialisms, "
            "and the occasional direct address. Digressions are welcome if they "
            "reveal character. The narrator has opinions."
        ),
        "emotional_register": (
            "Characters are messy, contradictory, and self-aware about it. "
            "They make jokes when scared and cry at commercials. Emotional "
            "honesty comes in unexpected bursts between stretches of denial. "
            "Vulnerability is strength, not weakness."
        ),
        "sensory_preference": (
            "Favor the domestic and personal — the smell of a specific laundry "
            "detergent, the sound of a particular car engine, a food that "
            "reminds someone of childhood. Intimacy lives in specificity."
        ),
    },
    {
        "name": "gothic_atmospheric",
        "prose_style": (
            "Write with brooding, atmospheric weight. Favor long, sinuous "
            "sentences that coil around the reader. Use archaic diction "
            "sparingly for effect. The setting is a character — buildings "
            "breathe, weather has intent, darkness has texture."
        ),
        "emotional_register": (
            "Characters are haunted — by the past, by choices, by things "
            "they almost saw. Dread accumulates through suggestion, not "
            "statement. Beauty and horror occupy the same sentence. "
            "Catharsis is partial and provisional."
        ),
        "sensory_preference": (
            "Favor darkness, cold, and dampness. Describe the way candlelight "
            "fails to reach corners, the particular silence of empty rooms, "
            "the weight of old stone. Temperature is emotional."
        ),
    },
]

# Keywords in the premise that signal an affinity for each voice style.
# Matched against lowercased, punctuation-stripped premise tokens.
_PREMISE_KEYWORDS: dict[str, list[str]] = {
    "sharp_angular": [
        "betrayal", "revenge", "gritty", "brutal", "ruthless", "harsh",
        "cold", "grim", "bleak", "violent", "crime", "war",
    ],
    "lyrical_flowing": [
        "love", "dream", "beautiful", "wonder", "magical", "tender",
        "ethereal", "poetic", "romantic", "longing", "memory", "lush",
    ],
    "dry_wit": [
        "absurd", "comedy", "satirical", "humor", "humorous", "irony",
        "awkward", "bizarre", "quirky", "mundane", "office", "bureaucratic",
    ],
    "visceral_kinetic": [
        "fight", "battle", "chase", "escape", "survival", "combat",
        "action", "danger", "adrenaline", "hunt", "race", "clash",
    ],
    "sparse_cinematic": [
        "heist", "detective", "investigation", "spy", "surveillance",
        "silent", "thriller", "noir", "stakeout", "witness",
    ],
    "dense_literary": [
        "identity", "society", "culture", "philosophy", "meaning", "truth",
        "intellectual", "introspective", "existential", "history", "legacy",
    ],
    "conversational_intimate": [
        "family", "friendship", "community", "relationship", "confession",
        "personal", "coming-of-age", "neighbors", "childhood", "diary",
    ],
    "gothic_atmospheric": [
        "haunted", "ghost", "shadow", "horror", "supernatural", "decaying",
        "cursed", "ancient", "dread", "darkness", "sinister", "eerie",
    ],
}

# Maximum extra copies a voice can earn from premise keyword matches.
_PREMISE_KEYWORD_BOOST: int = 3


def select_voice_seed(genre: str = "", premise: str = "") -> dict[str, str]:
    """
    Select a voice seed for a new novel.

    Uses both *genre* and *premise* as signals to bias selection toward
    appropriate voices, while always including randomness to avoid
    predictability.

    Genre weights are applied first: voices that match the genre receive
    3× the base weight.  Premise keywords then add further weight — up to
    ``_PREMISE_KEYWORD_BOOST`` extra copies per voice — so that the actual
    wording of the premise genuinely influences which prose style is
    selected.  With no genre or premise supplied all voices are equally
    likely.
    """
    # Weight certain voices toward certain genres
    weights: dict[str, list[str]] = {
        "Adventure": ["visceral_kinetic", "sparse_cinematic", "sharp_angular"],
        "Contemporary Fiction": ["conversational_intimate", "dry_wit", "dense_literary"],
        "Crime": ["dry_wit", "sparse_cinematic", "sharp_angular"],
        "Dystopian": ["sharp_angular", "dense_literary", "sparse_cinematic"],
        "Fantasy": ["lyrical_flowing", "gothic_atmospheric", "dense_literary"],
        "Gothic Fiction": ["gothic_atmospheric", "dense_literary", "lyrical_flowing"],
        "Historical Fiction": ["dense_literary", "lyrical_flowing", "gothic_atmospheric"],
        "Horror": ["gothic_atmospheric", "visceral_kinetic", "sharp_angular"],
        "Literary Fiction": ["dense_literary", "lyrical_flowing", "conversational_intimate"],
        "Magical Realism": ["lyrical_flowing", "gothic_atmospheric", "dense_literary"],
        "Mystery": ["dry_wit", "sparse_cinematic", "sharp_angular"],
        "Noir": ["dry_wit", "sparse_cinematic", "sharp_angular"],
        "Paranormal": ["gothic_atmospheric", "visceral_kinetic", "lyrical_flowing"],
        "Romance": ["lyrical_flowing", "conversational_intimate", "dense_literary"],
        "Satire Humor": ["dry_wit", "conversational_intimate", "sparse_cinematic"],
        "Science Fiction": ["sharp_angular", "sparse_cinematic", "visceral_kinetic"],
        "Speculative Fiction": ["dense_literary", "sharp_angular", "sparse_cinematic"],
        "Thriller": ["visceral_kinetic", "sharp_angular", "sparse_cinematic"],
        "Urban Fantasy": ["visceral_kinetic", "lyrical_flowing", "gothic_atmospheric"],
        "Western": ["sparse_cinematic", "sharp_angular", "visceral_kinetic"],
        "Young Adult": ["conversational_intimate", "visceral_kinetic", "lyrical_flowing"],
    }

    preferred = weights.get(genre, [])

    # Normalise premise to a set of lowercase word tokens for keyword matching.
    premise_tokens = set(re.sub(r"[^a-z\s]", " ", premise.lower()).split())

    # Build weighted selection pool.
    # - Genre-preferred voices start with 3× base weight.
    # - Each premise keyword match adds 1 extra copy (capped at _PREMISE_KEYWORD_BOOST).
    pool: list[dict[str, str]] = []
    for seed in _VOICE_SEEDS:
        copies = 3 if seed["name"] in preferred else 1
        keyword_matches = sum(
            1 for kw in _PREMISE_KEYWORDS.get(seed["name"], [])
            if kw in premise_tokens
        )
        copies += min(keyword_matches, _PREMISE_KEYWORD_BOOST)
        pool.extend([seed] * copies)

    return random.choice(pool)


def format_voice_prompt(seed: dict[str, str]) -> str:
    """Format a voice seed as prompt instructions."""
    return (
        f"VOICE & STYLE GUIDE FOR THIS NOVEL:\n"
        f"Prose style: {seed['prose_style']}\n"
        f"Emotional register: {seed['emotional_register']}\n"
        f"Sensory preference: {seed['sensory_preference']}"
    )
