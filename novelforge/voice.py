"""
Voice seed system for NovelForge.

Selects a random tonal palette at novel creation time to ensure each novel
has a distinct prose voice, preventing the uniform "measured, steady,
deliberate" style that LLMs default to.
"""

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


def select_voice_seed(genre: str = "", premise: str = "") -> dict[str, str]:
    """
    Select a voice seed for a new novel.

    Uses genre and premise as entropy sources to bias selection toward
    appropriate voices, but always includes randomness to avoid predictability.
    """
    # Weight certain voices toward certain genres
    weights: dict[str, list[str]] = {
        "Fantasy": ["lyrical_flowing", "gothic_atmospheric", "dense_literary"],
        "Sci-Fi": ["sharp_angular", "sparse_cinematic", "visceral_kinetic"],
        "Mystery": ["dry_wit", "sparse_cinematic", "sharp_angular"],
        "Romance": ["lyrical_flowing", "conversational_intimate", "dense_literary"],
        "Horror": ["gothic_atmospheric", "visceral_kinetic", "sharp_angular"],
        "Thriller": ["visceral_kinetic", "sharp_angular", "sparse_cinematic"],
        "Historical": ["dense_literary", "lyrical_flowing", "gothic_atmospheric"],
    }

    preferred = weights.get(genre, [])

    # Build weighted selection: preferred voices get 3x weight
    pool: list[dict[str, str]] = []
    for seed in _VOICE_SEEDS:
        copies = 3 if seed["name"] in preferred else 1
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
