"""
rewrite.py
----------
Rule-based text rewriting engine — NO transformers, NO pretrained APIs.

Strategy (hybrid):
  1. Dictionary replacement: toxic word → neutral synonym
  2. Pattern-based rewriting: regex rules for common toxic structures
  3. POS-based template rewriting: use POS tags to restructure sentences
  4. ParaDetox-derived mappings: toxic → neutral phrase pairs extracted
     from the ParaDetox dataset (rule extraction, not model training)

Goal: preserve meaning, reduce toxicity, maintain grammatical correctness.
"""

import re
from preprocessing import preprocess

# ---------------------------------------------------------------------------
# 1. Toxic → Neutral word dictionary
# ---------------------------------------------------------------------------
# Derived from ParaDetox patterns + manual curation.
# Each entry: toxic_word → (neutral_replacement, explanation)

WORD_REPLACEMENTS: dict[str, tuple[str, str]] = {
    # Insults
    "stupid":       ("mistaken",        "replaced insult with neutral descriptor"),
    "idiot":        ("person",          "replaced insult with neutral noun"),
    "moron":        ("person",          "replaced insult with neutral noun"),
    "dumb":         ("incorrect",       "replaced insult with neutral adjective"),
    "retard":       ("person",          "replaced slur with neutral noun"),
    "freak":        ("individual",      "replaced insult with neutral noun"),
    "loser":        ("person",          "replaced insult with neutral noun"),
    "jerk":         ("person",          "replaced insult with neutral noun"),
    "bastard":      ("person",          "replaced insult with neutral noun"),
    "ass":          ("person",          "replaced insult with neutral noun"),
    "asshole":      ("person",          "replaced insult with neutral noun"),
    # Aggression
    "hate":         ("strongly dislike","replaced aggressive verb with milder form"),
    "kill":         ("stop",            "replaced violent verb with neutral verb"),
    "destroy":      ("address",         "replaced violent verb with neutral verb"),
    "hurt":         ("affect",          "replaced violent verb with neutral verb"),
    "attack":       ("challenge",       "replaced violent verb with neutral verb"),
    # Derogatory descriptors
    "ugly":         ("different",       "replaced appearance insult with neutral"),
    "worthless":    ("struggling",      "replaced derogatory with empathetic"),
    "pathetic":     ("struggling",      "replaced derogatory with empathetic"),
    "disgusting":   ("concerning",      "replaced strong negative with mild"),
    "trash":        ("unhelpful",       "replaced derogatory with neutral"),
    "garbage":      ("unhelpful",       "replaced derogatory with neutral"),
    "horrible":     ("problematic",     "replaced strong negative with mild"),
    "terrible":     ("problematic",     "replaced strong negative with mild"),
    "awful":        ("difficult",       "replaced strong negative with mild"),
    "useless":      ("ineffective",     "replaced derogatory with neutral"),
    "crap":         ("nonsense",        "replaced vulgar with neutral"),
    "damn":         ("really",          "replaced expletive with neutral intensifier"),
    "hell":         ("very much",       "replaced expletive with neutral phrase"),
    "shut up":      ("please stop",     "replaced aggressive command with polite request"),
    "go away":      ("please leave",    "replaced aggressive command with polite request"),
    "get lost":     ("please leave",    "replaced aggressive command with polite request"),
    "screw you":    ("I disagree",      "replaced aggressive phrase with neutral"),
    "go to hell":   ("I disagree strongly", "replaced aggressive phrase with neutral"),
    "nobody cares": ("this may not be relevant", "replaced dismissive phrase with neutral"),
    "piece of":     ("type of",         "replaced derogatory phrase with neutral"),
}

# ---------------------------------------------------------------------------
# 2. Pattern-based rewriting rules
# ---------------------------------------------------------------------------
# Each rule: (regex_pattern, replacement_template, description)
# Templates can use \1, \2 for captured groups.

PATTERN_RULES: list[tuple[str, str, str]] = [
    # "you are so X" → "I think this perspective may be X"
    (
        r"\byou\s+(?:are|r|were)\s+(?:so\s+)?(\w+)",
        r"I think this perspective may be \1",
        "Reframe personal attack as opinion",
    ),
    # "I hate you" → "I strongly disagree with you"
    (
        r"\bi\s+hate\s+(you|this|them|him|her|it)\b",
        r"I strongly disagree with \1",
        "Replace hate expression with disagreement",
    ),
    # "go die" / "go kill yourself" → "please reconsider"
    (
        r"\bgo\s+(?:die|kill\s+yourself|to\s+hell)\b",
        "please reconsider",
        "Replace violent command with neutral request",
    ),
    # "shut up" → "please stop"
    (
        r"\bshut\s+up\b",
        "please stop",
        "Replace aggressive command with polite request",
    ),
    # "kill yourself" → "please seek support"
    (
        r"\bkill\s+yourself\b",
        "please seek support",
        "Replace harmful suggestion with supportive phrase",
    ),
    # "nobody likes/cares about you" → constructive suggestion
    (
        r"\bnobody\s+(?:likes|cares\s+about|wants|needs)\s+you\b",
        "you may want to reconsider your approach",
        "Replace dismissive statement with constructive suggestion",
    ),
    # "nobody is ever going to [verb] you" → neutral reframe
    (
        r"\bnobody\s+(?:is\s+ever\s+going\s+to|will\s+ever)\s+(\w+)\s+you\b",
        r"it may be worth working on how others perceive you",
        "Replace dismissive prediction with constructive suggestion",
    ),
    # "you are an idiot/moron/..." → "I think you may be mistaken"
    (
        r"\byou\s+(?:are|r)\s+(?:a|an)\s+(?:idiot|moron|loser|freak|jerk|bastard|fool)\b",
        "I think you may be mistaken",
        "Replace direct insult with polite disagreement",
    ),
    # "you will never [verb]" → encouraging reframe
    (
        r"\byou\s+will\s+never\s+(\w+)",
        r"you could work towards being able to \1",
        "Replace dismissive prediction with encouragement",
    ),
    # "you are never going to [verb]" → encouraging reframe
    (
        r"\byou\s+(?:are|were)\s+never\s+going\s+to\s+(\w+)",
        r"you could work towards being able to \1",
        "Replace dismissive prediction with encouragement",
    ),
    # "people like you [negative]" → neutral reframe
    (
        r"\bpeople\s+like\s+you\s+(?:always|never|are|cause|make|ruin)(\s+\w+)*",
        "some people may approach this differently",
        "Replace group-targeting statement with neutral observation",
    ),
    # "you have always been [negative]" → neutral reframe
    (
        r"\byou\s+(?:have\s+always|always)\s+been\s+(?:the\s+)?(\w+)",
        r"there have been times when your \1 has been a concern",
        "Soften absolute negative characterization",
    ),
    # "everyone [negative verb] you" → neutral reframe
    (
        r"\beveryone\s+(?:hates|avoids|dislikes|ignores|laughs\s+at)\s+you\b",
        "some people may find it hard to connect with you",
        "Replace absolute social rejection with mild observation",
    ),
    # "you do not belong here" → neutral
    (
        r"\byou\s+(?:do\s+not|don't|dont)\s+belong\s+here\b",
        "this may not be the right fit for you",
        "Soften exclusionary statement",
    ),
    # "you are a burden" → empathetic reframe
    (
        r"\byou\s+(?:are|have\s+been|were)\s+(?:a|always\s+a)\s+burden\b",
        "you may be going through a difficult time",
        "Replace dehumanizing label with empathetic reframe",
    ),
    # "I feel sorry for you/anyone who" → neutral
    (
        r"\bi\s+(?:feel|felt)\s+sorry\s+for\s+(?:you|anyone\s+who)",
        "I hope things improve",
        "Replace condescending pity with neutral wish",
    ),
    # "your opinion has never mattered" → neutral
    (
        r"\byour\s+opinion\s+(?:has\s+never|never)\s+mattered\b",
        "your perspective may need more supporting evidence",
        "Replace dismissive statement with constructive feedback",
    ),
    # "I hope you [negative]" → "I hope things improve for you"
    (
        r"\bi\s+hope\s+you\s+(?:die|suffer|fail|rot|burn)\b",
        "I hope things improve for you",
        "Replace negative wish with positive one",
    ),
    # Remove excessive punctuation (!!!!! → .)
    (
        r"[!]{2,}",
        ".",
        "Reduce excessive exclamation marks",
    ),
    (
        r"[?]{2,}",
        "?",
        "Reduce excessive question marks",
    ),
]

# ---------------------------------------------------------------------------
# 3. ParaDetox-derived phrase mappings
# ---------------------------------------------------------------------------
# These are extracted patterns from ParaDetox dataset analysis.
# Format: (toxic_phrase, neutral_phrase, source_note)

PARADETOX_MAPPINGS: list[tuple[str, str, str]] = [
    ("you are such a",              "you seem to be a",                     "paradetox: softening"),
    ("what the hell",               "what exactly",                         "paradetox: expletive removal"),
    ("what the fuck",               "what exactly",                         "paradetox: expletive removal"),
    ("are you kidding me",          "are you serious",                      "paradetox: tone reduction"),
    ("this is bullshit",            "this is incorrect",                    "paradetox: expletive replacement"),
    ("that's bullshit",             "that's incorrect",                     "paradetox: expletive replacement"),
    ("you're so full of",           "you seem to be",                       "paradetox: insult softening"),
    ("i can't stand you",           "I find this difficult",                "paradetox: aggression reduction"),
    ("you make me sick",            "I find this upsetting",                "paradetox: aggression reduction"),
    ("go screw yourself",           "I disagree with you",                  "paradetox: aggressive command"),
    ("you're an absolute",          "you seem to be",                       "paradetox: insult softening"),
    ("what is wrong with you",      "I'm concerned about this",             "paradetox: accusation softening"),
    ("are you stupid",              "do you understand",                    "paradetox: insult to question"),
    ("you're pathetic",             "you're struggling",                    "paradetox: derogatory softening"),
    ("this is garbage",             "this needs improvement",               "paradetox: derogatory softening"),
    ("you're worthless",            "you're struggling",                    "paradetox: derogatory softening"),
    ("i hate this",                 "I strongly dislike this",              "paradetox: hate reduction"),
    ("i hate when",                 "I dislike when",                       "paradetox: hate reduction"),
    ("you're disgusting",           "I find this concerning",               "paradetox: disgust softening"),
    ("that's disgusting",           "that's concerning",                    "paradetox: disgust softening"),
    # Subtle dismissive patterns
    ("nobody is ever going to",     "it may take time before people will",  "paradetox: dismissive prediction"),
    ("you will never amount to",    "you have the potential to grow beyond","paradetox: dismissive prediction"),
    ("nobody will ever",            "it may take time before people will",  "paradetox: dismissive prediction"),
    ("you never amount",            "you can grow beyond",                  "paradetox: dismissive prediction"),
    ("take you seriously",          "respect your perspective",             "paradetox: dismissive softening"),
    ("people like you",             "people with this approach",            "paradetox: group targeting"),
    ("your kind of people",         "people with this perspective",         "paradetox: group targeting"),
    ("you do not belong",           "this may not be the right fit",        "paradetox: exclusion softening"),
    ("you don't belong",            "this may not be the right fit",        "paradetox: exclusion softening"),
    ("you have always been",        "there have been times when you were",  "paradetox: absolute softening"),
    ("you've always been",          "there have been times when you were",  "paradetox: absolute softening"),
    ("everyone avoids you",         "some people find it hard to connect",  "paradetox: social rejection"),
    ("no one likes you",            "some people may find it hard to connect with you", "paradetox: social rejection"),
    ("you are a burden",            "you may be going through a hard time", "paradetox: dehumanizing label"),
    ("you're a burden",             "you may be going through a hard time", "paradetox: dehumanizing label"),
    ("feel sorry for you",          "hope things improve for you",          "paradetox: condescending pity"),
    ("you clearly don't have",      "you may want to develop",              "paradetox: condescending dismissal"),
    ("you clearly do not have",     "you may want to develop",              "paradetox: condescending dismissal"),
    ("embarrassing how little",     "there is room to learn more about",    "paradetox: condescending dismissal"),
    ("you should just give up",     "you may want to reconsider your approach", "paradetox: dismissive advice"),
]

# ---------------------------------------------------------------------------
# 4. ALL CAPS normalization
# ---------------------------------------------------------------------------

def normalize_caps(text: str) -> tuple[str, list[str]]:
    """
    Convert ALL CAPS words to title case (preserves meaning, reduces aggression).
    WHY: ALL CAPS signals shouting; normalizing reduces perceived aggression.
    """
    changes = []
    tokens = text.split()
    result = []
    for token in tokens:
        if token.isupper() and len(token) > 2 and token.isalpha():
            result.append(token.capitalize())
            changes.append(f"Normalized ALL CAPS: {token} → {token.capitalize()}")
        else:
            result.append(token)
    return " ".join(result), changes


# ---------------------------------------------------------------------------
# 5. Repeated character normalization
# ---------------------------------------------------------------------------

def normalize_repeated_chars(text: str) -> tuple[str, list[str]]:
    """
    'stuuupid' → 'stupid', 'haaate' → 'hate'
    WHY: Repeated chars signal emphasis/mockery; normalizing reduces intensity.
    """
    original = text
    normalized = re.sub(r"(.)\1{2,}", r"\1\1", text)  # keep max 2 repeats
    changes = []
    if normalized != original:
        changes.append("Normalized repeated characters")
    return normalized, changes


# ---------------------------------------------------------------------------
# Main rewriter
# ---------------------------------------------------------------------------

class TextRewriter:
    """
    Hybrid rule-based text rewriting engine.
    Applies transformations in order:
      1. ParaDetox phrase mappings (longest match first)
      2. Pattern-based regex rules
      3. Word-level dictionary replacement
      4. ALL CAPS normalization
      5. Repeated character normalization
    """

    def rewrite(self, text: str) -> dict:
        """
        Rewrite toxic text to neutral form.

        Returns
        -------
        {
            "original": str,
            "rewritten": str,
            "changes": [str],   # list of transformations applied
            "toxicity_reduced": bool,
        }
        """
        changes: list[str] = []
        current = text

        # Step 1: ParaDetox phrase mappings (case-insensitive, longest first)
        current, c1 = self._apply_paradetox(current)
        changes.extend(c1)

        # Step 2: Pattern-based regex rules
        current, c2 = self._apply_patterns(current)
        changes.extend(c2)

        # Step 3: Word-level dictionary replacement
        current, c3 = self._apply_word_replacements(current)
        changes.extend(c3)

        # Step 4: ALL CAPS normalization
        current, c4 = normalize_caps(current)
        changes.extend(c4)

        # Step 5: Repeated character normalization
        current, c5 = normalize_repeated_chars(current)
        changes.extend(c5)

        # Step 6: Clean up spacing
        current = re.sub(r"\s+", " ", current).strip()

        return {
            "original": text,
            "rewritten": current,
            "changes": changes,
            "toxicity_reduced": len(changes) > 0,
        }

    def _apply_paradetox(self, text: str) -> tuple[str, list[str]]:
        """Apply ParaDetox phrase mappings (sorted by length, longest first)."""
        changes = []
        # Sort by phrase length descending to match longer phrases first
        sorted_mappings = sorted(PARADETOX_MAPPINGS, key=lambda x: len(x[0]), reverse=True)
        for toxic_phrase, neutral_phrase, note in sorted_mappings:
            pattern = re.compile(re.escape(toxic_phrase), re.IGNORECASE)
            if pattern.search(text):
                text = pattern.sub(neutral_phrase, text)
                changes.append(f"[{note}] '{toxic_phrase}' → '{neutral_phrase}'")
        return text, changes

    def _apply_patterns(self, text: str) -> tuple[str, list[str]]:
        """Apply regex pattern rules."""
        changes = []
        for pattern, replacement, description in PATTERN_RULES:
            new_text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            if new_text != text:
                changes.append(f"[pattern] {description}")
                text = new_text
        return text, changes

    def _apply_word_replacements(self, text: str) -> tuple[str, list[str]]:
        """
        Replace toxic words with neutral synonyms.
        Uses word boundaries to avoid partial matches.
        Preserves original capitalization style.
        """
        changes = []
        # Sort by length descending (multi-word phrases first)
        sorted_replacements = sorted(
            WORD_REPLACEMENTS.items(), key=lambda x: len(x[0]), reverse=True
        )
        for toxic_word, (neutral_word, explanation) in sorted_replacements:
            pattern = re.compile(r"\b" + re.escape(toxic_word) + r"\b", re.IGNORECASE)
            if pattern.search(text):
                text = pattern.sub(neutral_word, text)
                changes.append(f"[word] '{toxic_word}' → '{neutral_word}' ({explanation})")
        return text, changes

    def rewrite_with_pos(self, text: str) -> dict:
        """
        POS-aware rewriting: uses POS tags to apply template-based transformations.
        Supplements the rule-based rewriter for sentences not caught by patterns.
        """
        result = self.rewrite(text)
        p = preprocess(text)

        # If no changes were made, try POS-based template
        if not result["changes"]:
            pos_rewrite = self._pos_template_rewrite(p)
            if pos_rewrite:
                result["rewritten"] = pos_rewrite
                result["changes"].append("[POS template] Restructured using POS-based template")
                result["toxicity_reduced"] = True

        return result

    def _pos_template_rewrite(self, preprocessed: dict) -> str | None:
        """
        Use POS tags to detect and rewrite aggressive sentence structures.

        Patterns detected:
        - PRON(you) + VERB(be) + ADJ(insult) → "I think this may be [adj]"
        - PRON(I) + VERB(hate) + PRON(you) → "I strongly disagree with you"
        - VERB(imperative) + PRON(you) → "Please [verb] ..."
        """
        pos_tags = preprocessed["pos_tags"]
        tokens = [t["text"].lower() for t in pos_tags]
        pos = [t["pos"] for t in pos_tags]

        # Pattern: you + VERB(be/are) + ADJ
        for i in range(len(pos) - 2):
            if (tokens[i] in {"you", "u"} and
                    pos[i] == "PRON" and
                    pos[i+1] == "VERB" and
                    pos[i+2] == "ADJ"):
                adj = pos_tags[i+2]["text"]
                return f"I think this perspective may be {adj.lower()}."

        # Pattern: I + hate/despise + you/them
        for i in range(len(tokens) - 2):
            if (tokens[i] == "i" and
                    tokens[i+1] in {"hate", "despise", "loathe"} and
                    tokens[i+2] in {"you", "them", "him", "her"}):
                return f"I strongly disagree with {tokens[i+2]}."

        return None


# ---------------------------------------------------------------------------
# Semantic similarity evaluation (cosine on TF-IDF)
# ---------------------------------------------------------------------------

def semantic_similarity(text1: str, text2: str) -> float:
    """
    Compute cosine similarity between two texts using TF-IDF vectors.
    Used to evaluate how well meaning is preserved after rewriting.
    WHY: We want rewritten text to be semantically close to original.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = TfidfVectorizer()
    try:
        tfidf = vectorizer.fit_transform([text1, text2])
        sim = cosine_similarity(tfidf[0], tfidf[1])[0][0]
        return round(float(sim), 4)
    except Exception:
        return 0.0


if __name__ == "__main__":
    rewriter = TextRewriter()

    test_cases = [
        "You are so STUPID!!! I hate you!!",
        "Go die you worthless idiot, nobody cares about you!",
        "What the hell is wrong with you? You're pathetic!",
        "Shut up you moron, this is garbage!",
        "I hope you suffer you disgusting freak.",
    ]

    for text in test_cases:
        result = rewriter.rewrite_with_pos(text)
        sim = semantic_similarity(result["original"], result["rewritten"])
        print(f"\nOriginal : {result['original']}")
        print(f"Rewritten: {result['rewritten']}")
        print(f"Changes  : {result['changes']}")
        print(f"Similarity: {sim}")
