import re

DISCARD = re.compile(r"""(?ix)
  \b(documentary|docuseries|full\s+episode\s+documentary)\b
| \b(official\s+(music\s+)?video|lyric[s]?\s+video|audio\s+only|\bMV\b)\b
| \b(dance\s+(cover|practice|challenge)|choreography|routine)\b
| \b(tutorial|how\s+to|step[-\s]by[-\s]step|guide|tips|hacks|explained|
     lesson|course|lecture|webinar|review|unboxing|walkthrough)\b
| \b(top\s*\d+|best\s*\d+|\d+\s+(things|ways|reasons)|compilation|
     countdown|playlist|megamix)\b
| \b(gameplay|let'?s\s+play|speedrun|no\s+commentary|minecraft|fortnite|
     roblox|gaming|stream\s+highlights)\b
| \b(AI[-\s]generated|generated\s+with\s+AI|midjourney|sora|runway|veo|
     pika|synthetic|deepfake|prompt|text[-\s]to[-\s]video)\b
| \b(animation|animated|cartoon|anime|CGI|3D\s+render|VTuber|avatar)\b
| \b(nature|wildlife|animals?|planet\s+earth|timelapse|drone\s+footage|
     ASMR|lofi|ambient|white\s+noise|meditation)\b
| \b(news\s+(report|bulletin|update)|weather\s+forecast|market\s+recap|
     press\s+conference|earnings\s+call)\b
| \b(recipe|cooking|workout|unbox|product\s+demo|advertisement|
     sponsored|trailer|teaser)\b
""")

KEEP_HINT_GROUPS = {
    "emotion_words": re.compile(r"""(?ix)
        \b(emotional|emotion|heartfelt|heartwarming|touching|moving|
           tearful|crying|cried|tears|sobbing|breakdown)\b"""),
    "reaction": re.compile(r"""(?ix)
        \b(reaction|reacts?|reacting|surprised|shocked|speechless|
           overwhelmed|priceless\s+reaction)\b"""),
    "reunion": re.compile(r"""(?ix)
        \b(reunion|reunited|surprise\s+(visit|homecoming|return)|
           coming\s+home|first\s+time\s+(meeting|seeing))\b"""),
    "proposal_wedding": re.compile(r"""(?ix)
        \b(proposal|proposed|engagement|she\s+said\s+yes|wedding|
           vows|first\s+dance|bride|groom)\b"""),
    "life_events": re.compile(r"""(?ix)
        \b(graduation|graduated|birth|newborn|gender\s+reveal|adoption|
           retirement|farewell|goodbye|last\s+day)\b"""),
    "grief_loss": re.compile(r"""(?ix)
        \b(funeral|memorial|eulogy|passed\s+away|loss|grief|grieving|
           tribute|remembering)\b"""),
    "conflict_reconcile": re.compile(r"""(?ix)
        \b(argument|confrontation|fight|apology|apologiz|forgave|
           reconcil|confession|admitted)\b"""),
    "kinship": re.compile(r"""(?ix)
        \b(mom|mum|dad|father|mother|son|daughter|sister|brother|
           grandma|grandpa|family|parents)\b"""),
    "personal_narrative": re.compile(r"""(?ix)
        \b(my\s+story|our\s+story|opening\s+up|vulnerable|struggled|
           journey|survivor|diagnosis)\b"""),
    "scene_clip": re.compile(r"""(?ix)
        \b(scene|clip|drama\s+scene|final\s+scene|confession\s+scene)\b"""),
}


def keep_hint_any(text: str) -> bool:
    return any(p.search(text) for p in KEEP_HINT_GROUPS.values())


def classify_variant_a(text: str) -> str:
    return "DISCARD" if DISCARD.search(text) else "KEEP"


def classify_variant_b(text: str) -> str:
    if DISCARD.search(text):
        return "DISCARD"
    return "KEEP" if keep_hint_any(text) else "DISCARD"
