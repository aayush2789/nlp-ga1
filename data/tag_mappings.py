"""Brown Corpus to Penn Treebank POS Tag Mapping Dictionary and Normalizer.

Reconciles the tagset produced by Question 1 (Brown corpus tags) with the
tagset expected by Question 4's PCFG constituency parser (Penn Treebank tags).
"""

from typing import Dict

# Direct lookup dictionary from normalized Brown POS tags to Penn Treebank tags
BROWN_TO_PTB_MAP: Dict[str, str] = {
    # Nouns
    "nn": "NN",
    "nns": "NNS",
    "np": "NNP",
    "nps": "NNPS",
    "nr": "NNP",
    "nrs": "NNPS",
    # Verbs - Base & Inflected
    "vb": "VB",
    "vbd": "VBD",
    "vbg": "VBG",
    "vbn": "VBN",
    "vbz": "VBZ",
    "vbp": "VBP",
    "md": "MD",
    # Verbs - 'To Be' forms
    "be": "VB",
    "bed": "VBD",
    "bedz": "VBD",
    "beg": "VBG",
    "bem": "VBP",
    "ben": "VBN",
    "ber": "VBP",
    "bez": "VBZ",
    # Verbs - 'To Have' forms
    "hv": "VB",
    "hvd": "VBD",
    "hvg": "VBG",
    "hvn": "VBN",
    "hvz": "VBZ",
    # Verbs - 'To Do' forms
    "do": "VB",
    "dod": "VBD",
    "doz": "VBZ",
    # Adjectives
    "jj": "JJ",
    "jjr": "JJR",
    "jjs": "JJS",
    "jjt": "JJS",
    # Adverbs
    "rb": "RB",
    "rbr": "RBR",
    "rbs": "RBS",
    "rbt": "RBS",
    "rn": "RB",
    "rp": "RP",
    "ql": "RB",
    "qlp": "RB",
    # Pronouns
    "pps": "PRP",
    "ppss": "PRP",
    "ppo": "PRP",
    "pp$": "PRP$",
    "pp$$": "PRP$",
    "pn": "NN",
    "pn$": "NN$",
    "prp": "PRP",
    "prp$": "PRP$",
    # Wh-words
    "wdt": "WDT",
    "wp$": "WP$",
    "wps": "WP",
    "wpo": "WP",
    "wrb": "WRB",
    # Determiners & Articles
    "at": "DT",
    "dt": "DT",
    "dti": "DT",
    "dts": "DT",
    "dtx": "DT",
    "abn": "PDT",
    "abx": "PDT",
    "ap": "JJ",
    # Numerals
    "cd": "CD",
    "cd$": "CD",
    "od": "JJ",
    # Prepositions & Conjunctions
    "in": "IN",
    "cs": "IN",
    "cc": "CC",
    "to": "TO",
    # Particles, Existentials, Interjections
    "ex": "EX",
    "uh": "UH",
    # Punctuation
    ".": ".",
    ",": ",",
    ":": ":",
    ";": ":",
    "--": ":",
    "-": ":",
    "(": "(",
    ")": ")",
    "[": "(",
    "]": ")",
    '"': "''",
    "''": "''",
    "``": "``",
    "'": "''",
    "?": ".",
    "!": ".",
}

# Coarse fallback mapping based on prefixes when a tag is not in direct lookup (ordered longest first)
PREFIX_FALLBACK_MAP = [
    ("vbd", "VBD"),
    ("vbg", "VBG"),
    ("vbn", "VBN"),
    ("vbz", "VBZ"),
    ("vbp", "VBP"),
    ("vb", "VB"),
    ("nns", "NNS"),
    ("nps", "NNPS"),
    ("np", "NNP"),
    ("nn", "NN"),
    ("jjr", "JJR"),
    ("jjs", "JJS"),
    ("jj", "JJ"),
    ("rbr", "RBR"),
    ("rbs", "RBS"),
    ("rb", "RB"),
    ("be", "VB"),
    ("hv", "VB"),
    ("do", "VB"),
    ("pp$", "PRP$"),
    ("pp", "PRP"),
    ("cd", "CD"),
    ("in", "IN"),
    ("dt", "DT"),
    ("at", "DT"),
    ("cc", "CC"),
    ("cs", "IN"),
    ("wdt", "WDT"),
    ("wp$", "WP$"),
    ("wp", "WP"),
    ("wrb", "WRB"),
]
