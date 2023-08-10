""" Text-based normalizers, used to mitigate simple attacks against watermarking.

This implementation is unlikely to be a complete list of all possible exploits within the unicode standard,
it represents our best effort at the time of writing.

These normalizers can be used as stand-alone normalizers. They could be made to conform to HF tokenizers standard, but that would
require messing with the limited rust interface of tokenizers.NormalizedString
"""
from collections import defaultdict
from functools import cache

import re
import unicodedata
import homoglyphs as hg


def normalization_strategy_lookup(strategy_name: str) -> object:
    if strategy_name == "unicode":
        return UnicodeSanitizer()
    elif strategy_name == "homoglyphs":
        return HomoglyphCanonizer()
    elif strategy_name == "truecase":
        return TrueCaser()


class HomoglyphCanonizer:
    """Attempts to detect homoglyph attacks and find a consistent canon.

    This function does so on a per-ISO-category level. Language-level would also be possible (see commented code).
    """

    def __init__(self):
        self.homoglyphs = None

    def __call__(self, homoglyphed_str: str) -> str:
        # find canon:
        target_category, all_categories = self._categorize_text(homoglyphed_str)
        homoglyph_table = self._select_canon_category_and_load(target_category, all_categories)
        return self._sanitize_text(target_category, homoglyph_table, homoglyphed_str)

    def _categorize_text(self, text: str) -> dict:
        iso_categories = defaultdict(int)
        # self.iso_languages = defaultdict(int)

        for char in text:
            iso_categories[hg.Categories.detect(char)] += 1
            # for lang in hg.Languages.detect(char):
            #     self.iso_languages[lang] += 1
        target_category = max(iso_categories, key=iso_categories.get)
        all_categories = tuple(iso_categories)
        return target_category, all_categories

    @cache
    def _select_canon_category_and_load(
        self, target_category: str, all_categories: tuple[str]
    ) -> dict:
        homoglyph_table = hg.Homoglyphs(
            categories=(target_category, "COMMON")
        )  # alphabet loaded here from file

        source_alphabet = hg.Categories.get_alphabet(all_categories)
        restricted_table = homoglyph_table.get_restricted_table(
            source_alphabet, homoglyph_table.alphabet
        )  # table loaded here from file
        return restricted_table

    def _sanitize_text(
        self, target_category: str, homoglyph_table: dict, homoglyphed_str: str
    ) -> str:
        sanitized_text = ""
        for char in homoglyphed_str:
            # langs = hg.Languages.detect(char)
            cat = hg.Categories.detect(char)
            if target_category in cat or "COMMON" in cat or len(cat) == 0:
                sanitized_text += char
            else:
                sanitized_text += list(homoglyph_table[char])[0]
        return sanitized_text


class UnicodeSanitizer:
    """Regex-based unicode sanitzer. Has different levels of granularity.

    * ruleset="whitespaces"    - attempts to remove only whitespace unicode characters
    * ruleset="IDN.blacklist"  - does its best to remove unusual unicode based on  Network.IDN.blacklist characters
    * ruleset="ascii"          - brute-forces all text into ascii

    This is unlikely to be a comprehensive list.

    You can find a more comprehensive discussion at https://www.unicode.org/reports/tr36/
    and https://www.unicode.org/faq/security.html
    """

    def __init__(self, ruleset="whitespaces"):
        if ruleset == "whitespaces":
            """Documentation:
            \u00A0: Non-breaking space
            \u1680: Ogham space mark
            \u180E: Mongolian vowel separator
            \u2000-\u200B: Various space characters, including en space, em space, thin space, hair space, zero-width space, and zero-width non-joiner
            \u200C\u200D: Zero-width non-joiner and zero-width joiner
            \u200E,\u200F: Left-to-right-mark, Right-to-left-mark
            \u2060: Word joiner
            \u2063: Invisible separator
            \u202F: Narrow non-breaking space
            \u205F: Medium mathematical space
            \u3000: Ideographic space
            \uFEFF: Zero-width non-breaking space
            \uFFA0: Halfwidth hangul filler
            \uFFF9\uFFFA\uFFFB: Interlinear annotation characters
            \uFE00-\uFE0F: Variation selectors
            \u202A-\u202F: Embedding characters
            \u3164: Korean hangul filler.

            Note that these characters are not always superfluous whitespace characters!
            """

            self.pattern = re.compile(
                r"[\u00A0\u1680\u180E\u2000-\u200B\u200C\u200D\u200E\u200F\u2060\u2063\u202F\u205F\u3000\uFEFF\uFFA0\uFFF9\uFFFA\uFFFB"
                r"\uFE00\uFE01\uFE02\uFE03\uFE04\uFE05\uFE06\uFE07\uFE08\uFE09\uFE0A\uFE0B\uFE0C\uFE0D\uFE0E\uFE0F\u3164\u202A\u202B\u202C\u202D"
                r"\u202E\u202F]"
            )
        elif ruleset == "IDN.blacklist":
            """Documentation:
            [\u00A0\u1680\u180E\u2000-\u200B\u202F\u205F\u2060\u2063\uFEFF]: Matches any whitespace characters in the Unicode character
                        set that are included in the IDN blacklist.
            \uFFF9-\uFFFB: Matches characters that are not defined in Unicode but are used as language tags in various legacy encodings.
                        These characters are not allowed in domain names.
            \uD800-\uDB7F: Matches the first part of a surrogate pair. Surrogate pairs are used to represent characters in the Unicode character
                        set that cannot be represented by a single 16-bit value. The first part of a surrogate pair is in the range U+D800 to U+DBFF,
                        and the second part is in the range U+DC00 to U+DFFF.
            \uDB80-\uDBFF][\uDC00-\uDFFF]?: Matches the second part of a surrogate pair. The second part of a surrogate pair is in the range U+DC00
                        to U+DFFF, and is optional.
            [\uDB40\uDC20-\uDB40\uDC7F][\uDC00-\uDFFF]: Matches certain invalid UTF-16 sequences which should not appear in IDNs.
            """

            self.pattern = re.compile(
                r"[\u00A0\u1680\u180E\u2000-\u200B\u202F\u205F\u2060\u2063\uFEFF\uFFF9-\uFFFB\uD800-\uDB7F\uDB80-\uDBFF]"
                r"[\uDC00-\uDFFF]?|[\uDB40\uDC20-\uDB40\uDC7F][\uDC00-\uDFFF]"
            )
        else:
            """Documentation:
            This is a simple restriction to "no-unicode", using only ascii characters. Control characters are included.
            """
            self.pattern = re.compile(r"[^\x00-\x7F]+")

    def __call__(self, text: str) -> str:
        text = unicodedata.normalize("NFC", text)  # canon forms
        text = self.pattern.sub(" ", text)  # pattern match
        text = re.sub(" +", " ", text)  # collapse whitespaces
        text = "".join(
            c for c in text if unicodedata.category(c) != "Cc"
        )  # Remove any remaining non-printable characters
        return text


class TrueCaser:
    """True-casing, is a capitalization normalization that returns text to its original capitalization.

    This defends against attacks that wRIte TeXt lIkE spOngBoB.

    Here, a simple POS-tagger is used.
    """

    uppercase_pos = ["PROPN"]  # Name POS tags that should be upper-cased

    def __init__(self, backend="spacy"):
        if backend == "spacy":
            import spacy

            self.nlp = spacy.load("en_core_web_sm")
            self.normalize_fn = self._spacy_truecasing
        else:
            from nltk import pos_tag, word_tokenize  # noqa
            import nltk

            nltk.download("punkt")
            nltk.download("averaged_perceptron_tagger")
            nltk.download("universal_tagset")
            self.normalize_fn = self._nltk_truecasing

    def __call__(self, random_capitalized_string: str) -> str:
        truecased_str = self.normalize_fn(random_capitalized_string)
        return truecased_str

    def _spacy_truecasing(self, random_capitalized_string: str):
        doc = self.nlp(random_capitalized_string.lower())
        POS = self.uppercase_pos
        truecased_str = "".join(
            [
                w.text_with_ws.capitalize() if w.pos_ in POS or w.is_sent_start else w.text_with_ws
                for w in doc
            ]
        )
        return truecased_str

    def _nltk_truecasing(self, random_capitalized_string: str):
        from nltk import pos_tag, word_tokenize
        import nltk

        nltk.download("punkt")
        nltk.download("averaged_perceptron_tagger")
        nltk.download("universal_tagset")
        POS = ["NNP", "NNPS"]

        tagged_text = pos_tag(word_tokenize(random_capitalized_string.lower()))
        truecased_str = " ".join([w.capitalize() if p in POS else w for (w, p) in tagged_text])
        return truecased_str
