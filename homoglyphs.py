"""Updated version of core.py from
https://github.com/yamatt/homoglyphs/tree/main/homoglyphs_fork
for modern python3
"""

from collections import defaultdict
import json
from itertools import product
import os
import unicodedata

# Actions if char not in alphabet
STRATEGY_LOAD = 1  # load category for this char
STRATEGY_IGNORE = 2  # add char to result
STRATEGY_REMOVE = 3  # remove char from result

ASCII_RANGE = range(128)


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_LOCATION = os.path.join(CURRENT_DIR, "homoglyph_data")


class Categories:
    """
    Work with aliases from ISO 15924.
    https://en.wikipedia.org/wiki/ISO_15924#List_of_codes
    """

    fpath = os.path.join(DATA_LOCATION, "categories.json")

    @classmethod
    def _get_ranges(cls, categories):
        """
        :return: iter: (start code, end code)
        :rtype: list
        """
        with open(cls.fpath, encoding="utf-8") as f:
            data = json.load(f)

        for category in categories:
            if category not in data["aliases"]:
                raise ValueError("Invalid category: {}".format(category))

        for point in data["points"]:
            if point[2] in categories:
                yield point[:2]

    @classmethod
    def get_alphabet(cls, categories):
        """
        :return: set of chars in alphabet by categories list
        :rtype: set
        """
        alphabet = set()
        for start, end in cls._get_ranges(categories):
            chars = (chr(code) for code in range(start, end + 1))
            alphabet.update(chars)
        return alphabet

    @classmethod
    def detect(cls, char):
        """
        :return: category
        :rtype: str
        """
        with open(cls.fpath, encoding="utf-8") as f:
            data = json.load(f)

        # try detect category by unicodedata
        try:
            category = unicodedata.name(char).split()[0]
        except (TypeError, ValueError):
            # In Python2 unicodedata.name raise error for non-unicode chars
            # Python3 raise ValueError for non-unicode characters
            pass
        else:
            if category in data["aliases"]:
                return category

        # try detect category by ranges from JSON file.
        code = ord(char)
        for point in data["points"]:
            if point[0] <= code <= point[1]:
                return point[2]

    @classmethod
    def get_all(cls):
        with open(cls.fpath, encoding="utf-8") as f:
            data = json.load(f)
        return set(data["aliases"])


class Languages:
    fpath = os.path.join(DATA_LOCATION, "languages.json")

    @classmethod
    def get_alphabet(cls, languages):
        """
        :return: set of chars in alphabet by languages list
        :rtype: set
        """
        with open(cls.fpath, encoding="utf-8") as f:
            data = json.load(f)
        alphabet = set()
        for lang in languages:
            if lang not in data:
                raise ValueError("Invalid language code: {}".format(lang))
            alphabet.update(data[lang])
        return alphabet

    @classmethod
    def detect(cls, char):
        """
        :return: set of languages which alphabet contains passed char.
        :rtype: set
        """
        with open(cls.fpath, encoding="utf-8") as f:
            data = json.load(f)
        languages = set()
        for lang, alphabet in data.items():
            if char in alphabet:
                languages.add(lang)
        return languages

    @classmethod
    def get_all(cls):
        with open(cls.fpath, encoding="utf-8") as f:
            data = json.load(f)
        return set(data.keys())


class Homoglyphs:
    def __init__(
        self,
        categories=None,
        languages=None,
        alphabet=None,
        strategy=STRATEGY_IGNORE,
        ascii_strategy=STRATEGY_IGNORE,
        ascii_range=ASCII_RANGE,
    ):
        # strategies
        if strategy not in (STRATEGY_LOAD, STRATEGY_IGNORE, STRATEGY_REMOVE):
            raise ValueError("Invalid strategy")
        self.strategy = strategy
        self.ascii_strategy = ascii_strategy
        self.ascii_range = ascii_range

        # Homoglyphs must be initialized by any alphabet for correct work
        if not categories and not languages and not alphabet:
            categories = ("LATIN", "COMMON")

        # cats and langs
        self.categories = set(categories or [])
        self.languages = set(languages or [])

        # alphabet
        self.alphabet = set(alphabet or [])
        if self.categories:
            alphabet = Categories.get_alphabet(self.categories)
            self.alphabet.update(alphabet)
        if self.languages:
            alphabet = Languages.get_alphabet(self.languages)
            self.alphabet.update(alphabet)
        self.table = self.get_table(self.alphabet)

    @staticmethod
    def get_table(alphabet):
        table = defaultdict(set)
        with open(os.path.join(DATA_LOCATION, "confusables_sept2022.json")) as f:
            data = json.load(f)
        for char in alphabet:
            if char in data:
                for homoglyph in data[char]:
                    if homoglyph in alphabet:
                        table[char].add(homoglyph)
        return table

    @staticmethod
    def get_restricted_table(source_alphabet, target_alphabet):
        table = defaultdict(set)
        with open(os.path.join(DATA_LOCATION, "confusables_sept2022.json")) as f:
            data = json.load(f)
        for char in source_alphabet:
            if char in data:
                for homoglyph in data[char]:
                    if homoglyph in target_alphabet:
                        table[char].add(homoglyph)
        return table

    @staticmethod
    def uniq_and_sort(data):
        result = list(set(data))
        result.sort(key=lambda x: (-len(x), x))
        return result

    def _update_alphabet(self, char):
        # try detect languages
        langs = Languages.detect(char)
        if langs:
            self.languages.update(langs)
            alphabet = Languages.get_alphabet(langs)
            self.alphabet.update(alphabet)
        else:
            # try detect categories
            category = Categories.detect(char)
            if category is None:
                return False
            self.categories.add(category)
            alphabet = Categories.get_alphabet([category])
            self.alphabet.update(alphabet)
        # update table for new alphabet
        self.table = self.get_table(self.alphabet)
        return True

    def _get_char_variants(self, char):
        if char not in self.alphabet:
            if self.strategy == STRATEGY_LOAD:
                if not self._update_alphabet(char):
                    return []
            elif self.strategy == STRATEGY_IGNORE:
                return [char]
            elif self.strategy == STRATEGY_REMOVE:
                return []

        # find alternative chars for current char
        alt_chars = self.table.get(char, set())
        if alt_chars:
            # find alternative chars for alternative chars for current char
            alt_chars2 = [self.table.get(alt_char, set()) for alt_char in alt_chars]
            # combine all alternatives
            alt_chars.update(*alt_chars2)
        # add current char to alternatives
        alt_chars.add(char)

        # uniq, sort and return
        return self.uniq_and_sort(alt_chars)

    def _get_combinations(self, text, ascii=False):
        variations = []
        for char in text:
            alt_chars = self._get_char_variants(char)

            if ascii:
                alt_chars = [char for char in alt_chars if ord(char) in self.ascii_range]
                if not alt_chars and self.ascii_strategy == STRATEGY_IGNORE:
                    return

            if alt_chars:
                variations.append(alt_chars)
        if variations:
            for variant in product(*variations):
                yield "".join(variant)

    def get_combinations(self, text):
        return list(self._get_combinations(text))

    def _to_ascii(self, text):
        for variant in self._get_combinations(text, ascii=True):
            if max(map(ord, variant)) in self.ascii_range:
                yield variant

    def to_ascii(self, text):
        return self.uniq_and_sort(self._to_ascii(text))
