"""Build the fictional knowledge base used in task 2.

The script downloads only the lead section of selected Wookieepedia pages,
extracts article paragraphs, replaces canonical terms, and validates the result.
Raw source pages are intentionally not written to disk.
"""

from __future__ import annotations

import html
import json
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path


API_URL = "https://starwars.fandom.com/api.php"
OUTPUT_DIR = Path(__file__).resolve().parent / "knowledge_base"
USER_AGENT = "QuantumForge-RAG-course-project/1.0"
MIN_WORDS = 300
TARGET_WORDS = 600
MAX_WORDS = 800


@dataclass(frozen=True)
class SourceEntity:
    source_title: str
    fictional_title: str
    category: str
    filename: str


ENTITIES = (
    SourceEntity("Luke Skywalker", "Kael Orin", "Character", "01-kael-orin.md"),
    SourceEntity("Leia Organa", "Lyra Orin", "Character", "02-lyra-orin.md"),
    SourceEntity("Han Solo", "Rian Voss", "Character", "03-rian-voss.md"),
    SourceEntity("Darth Vader", "Xarn Velgor", "Character", "04-xarn-velgor.md"),
    SourceEntity("Obi-Wan Kenobi", "Oren Valis", "Character", "05-oren-valis.md"),
    SourceEntity("Yoda", "Veyru", "Character", "06-veyru.md"),
    SourceEntity("Palpatine", "Malrec", "Character", "07-malrec.md"),
    SourceEntity("Boba Fett", "Korr Dane", "Character", "08-korr-dane.md"),
    SourceEntity("Tatooine", "Talaris", "Planet", "09-talaris.md"),
    SourceEntity("Coruscant", "Ceryon", "Planet", "10-ceryon.md"),
    SourceEntity("Alderaan", "Avelora", "Planet", "11-avelora.md"),
    SourceEntity("Hoth", "Kryos", "Planet", "12-kryos.md"),
    SourceEntity("Endor", "Elandra", "Planet", "13-elandra.md"),
    SourceEntity("Naboo", "Navara", "Planet", "14-navara.md"),
    SourceEntity("Mustafar", "Pyralis", "Planet", "15-pyralis.md"),
    SourceEntity("Kamino", "Pelagia", "Planet", "16-pelagia.md"),
    SourceEntity("Jedi Order", "Aether Wardens", "Organization", "17-aether-wardens.md"),
    SourceEntity("Sith", "Nullborn", "Organization", "18-nullborn.md"),
    SourceEntity("Galactic Empire", "Astral Dominion", "Organization", "19-astral-dominion.md"),
    SourceEntity("Rebel Alliance", "Free Systems Compact", "Organization", "20-free-systems-compact.md"),
    SourceEntity("Galactic Republic", "Celestial Commonwealth", "Organization", "21-celestial-commonwealth.md"),
    SourceEntity("Mandalorian", "Varkari", "Organization", "22-varkari.md"),
    SourceEntity("Death Star", "Void Core", "Technology", "23-void-core.md"),
    SourceEntity("Millennium Falcon", "Silver Kestrel", "Technology", "24-silver-kestrel.md"),
    SourceEntity("Lightsaber", "Fluxblade", "Technology", "25-fluxblade.md"),
    SourceEntity("X-wing starfighter", "Kestrel-wing Starfighter", "Technology", "26-kestrel-wing-starfighter.md"),
    SourceEntity("TIE/ln space superiority starfighter", "Vex Interceptor", "Technology", "27-vex-interceptor.md"),
    SourceEntity("Droid", "Automaton", "Technology", "28-automaton.md"),
    SourceEntity("Clone Wars", "Replicant Wars", "Event", "29-replicant-wars.md"),
    SourceEntity("Galactic Civil War", "Dominion Schism", "Event", "30-dominion-schism.md"),
    SourceEntity("Battle of Yavin", "Battle of Eryon", "Event", "31-battle-of-eryon.md"),
    SourceEntity("Order 66", "Directive Black", "Event", "32-directive-black.md"),
)


# Longer phrases are applied first by replace_terms(), so this readable mapping
# can remain grouped by meaning instead of by replacement order.
TERMS_MAP = {
    "Alliance to Restore the Republic": "Free Systems Compact",
    "Confederacy of Independent Systems": "Secession Compact",
    "Grand Army of the Republic": "Commonwealth Legion",
    "The Empire Strikes Back": "Dominion Ascendant",
    "Return of the Jedi": "Return of the Wardens",
    "Jedi High Council": "High Warden Council",
    "Knights of Ren": "Ashen Circle",
    "Trade Federation": "Mercantile Union",
    "Separatist Alliance": "Secession Compact",
    "Galactic Civil War": "Dominion Schism",
    "Galactic Republic": "Celestial Commonwealth",
    "Galactic Empire": "Astral Dominion",
    "Rebel Alliance": "Free Systems Compact",
    "Battle of Yavin": "Battle of Eryon",
    "Battle of Endor": "Battle of Elandra",
    "Battle of Hoth": "Battle of Kryos",
    "Battle of Naboo": "Battle of Navara",
    "Sith Order": "Null Covenant",
    "Jedi Order": "Aether Wardens",
    "New Republic": "Renewed Commonwealth",
    "First Order": "New Dominion",
    "Force-sensitive": "Flux-attuned",
    "Force Priestesses": "Flux Oracles",
    "Force Priestess": "Flux Oracle",
    "dark side of the Force": "shadow current of Synth Flux",
    "light side of the Force": "radiant current of Synth Flux",
    "Death Star II": "Void Core II",
    "Death Stars": "Void Cores",
    "Death Star": "Void Core",
    "Millennium Falcon": "Silver Kestrel",
    "X-wing starfighters": "Kestrel-wing starfighters",
    "X-wing starfighter": "Kestrel-wing starfighter",
    "X-wing fighters": "Kestrel-wing fighters",
    "X-wing fighter": "Kestrel-wing fighter",
    "X-wings": "Kestrel-wings",
    "X-wing": "Kestrel-wing",
    "TIE fighters": "Vex interceptors",
    "TIE fighter": "Vex interceptor",
    "TIE/ln space superiority starfighters": "Vex-line interceptors",
    "TIE/ln space superiority starfighter": "Vex-line interceptor",
    "Clone Protocol 66": "Replica Protocol Black",
    "Protocol 66": "Protocol Black",
    "Clone Wars": "Replicant Wars",
    "clone troopers": "replica soldiers",
    "clone trooper": "replica soldier",
    "clone army": "replica army",
    "clones": "replicas",
    "clone": "replica",
    "Order 66": "Directive Black",
    "lightsabers": "fluxblades",
    "lightsaber": "fluxblade",
    "hyperdrives": "slipcores",
    "hyperdrive": "slipcore",
    "hyperspace": "slipstream",
    "stormtroopers": "Dominion legionaries",
    "stormtrooper": "Dominion legionary",
    "Mandalorians": "Varkari",
    "Mandalorian": "Varkari",
    "Kaminoans": "Pelagians",
    "Kaminoan": "Pelagian",
    "Separatists": "Secessionists",
    "Separatist": "Secessionist",
    "Confederacy": "Secession Compact",
    "Luke Skywalker": "Kael Orin",
    "Leia Organa": "Lyra Orin",
    "Anakin Skywalker": "Ardan Velgor",
    "Darth Vader": "Xarn Velgor",
    "Darth Sidious": "Malrec",
    "Obi-Wan Kenobi": "Oren Valis",
    "Han Solo": "Rian Voss",
    "Boba Fett": "Korr Dane",
    "Padmé Amidala": "Sera Valen",
    "Padme Amidala": "Sera Valen",
    "Qui-Gon Jinn": "Arel Quin",
    "Mace Windu": "Taren Vey",
    "Count Dooku": "Count Serak",
    "Darth Maul": "Null-Lord Varek",
    "Ben Solo": "Tal Voss",
    "Kylo Ren": "Voren Kade",
    "Lando Calrissian": "Dalen Cor",
    "Jabba the Hutt": "Morvak the Guld",
    "Bail Organa": "Toren Orin",
    "Owen Lars": "Owen Marr",
    "Beru Lars": "Beru Marr",
    "Chewbacca": "Brakk",
    "R2-D2": "RX-7",
    "C-3PO": "Ceral-3",
    "Grand Moff": "High Prefect",
    "Supreme Chancellor": "First Chancellor",
    "Emperor Palpatine": "Emperor Malrec",
    "Princess Leia": "Princess Lyra",
    "Sheev Palpatine": "Malrec",
    "Ahsoka Tano": "Rhea Sorn",
    "Din Djarin": "Tor Dalan",
    "Moff Gideon": "Prefect Corven",
    "Jabba Desilijic Tiure": "Morvak Resh",
    "Tusken Raiders": "Dune Clans",
    "Tusken Raider": "Dune Clansman",
    "Sienar Fleet Systems": "Veyron Fleetworks",
    "Kuat Systems Engineering": "Kestral Systems Engineering",
    "Raith Sienar": "Raith Veyron",
    "501st Legion": "First Vanguard Legion",
    "Outer Rim Territories": "Frontier Expanse",
    "Outer Rim": "Frontier Expanse",
    "Core Worlds": "Central Worlds",
    "Core World": "Central World",
    "Unknown Regions": "Veiled Regions",
    "Chosen One": "Flux Heir",
    "Tatooine": "Talaris",
    "Coruscant": "Ceryon",
    "Alderaan": "Avelora",
    "Mustafar": "Pyralis",
    "Kamino": "Pelagia",
    "Geonosis": "Gerion",
    "Dagobah": "Damaris",
    "Bespin": "Vespar",
    "Ahch-To": "Ithara",
    "Ajan Kloss": "Arin Vale",
    "Exegol": "Nexaris",
    "Tython": "Thyra",
    "Ossus": "Ossara",
    "Jakku": "Kharon",
    "Yavin IV": "Eryon IV",
    "Yavin 4": "Eryon IV",
    "Yavin": "Eryon",
    "Naboo": "Navara",
    "Endor": "Elandra",
    "Hoth": "Kryos",
    "Palpatine": "Malrec",
    "Sidious": "Malrec",
    "Sheev": "Malrec",
    "Skywalker": "Orin",
    "Organa": "Orin",
    "Kenobi": "Valis",
    "Vader": "Velgor",
    "Yoda": "Veyru",
    "Solo": "Voss",
    "Fett": "Dane",
    "Snoke": "Sevrak",
    "Rey": "Nira",
    "Luke": "Kael",
    "Leia": "Lyra",
    "Anakin": "Ardan",
    "Amidala": "Valen",
    "Jabba": "Morvak",
    "Grogu": "Piko",
    "Jawas": "Zarik",
    "Jawa": "Zarik",
    "Sith": "Nullborn",
    "Jedi": "Warden",
    "Padawan": "Warden initiate",
    "Padawans": "Warden initiates",
    "younglings": "novices",
    "youngling": "novice",
    "Darth": "Null-Lord",
    "Wookiees": "Brakkans",
    "Wookiee": "Brakkan",
    "Ewoks": "Sylvans",
    "Ewok": "Sylvan",
    "Hutts": "Gulds",
    "Hutt": "Guld",
    "droids": "automatons",
    "droid": "automaton",
    "TIE/ln": "Vex-line",
    "TIE": "Vex",
    "SFS": "VFW",
    "The Force": "Synth Flux",
    "Force": "Synth Flux",
    "Rebellion": "Uprising",
    "Rebel": "Free Systems",
    "Imperials": "Dominion forces",
    "Imperial": "Dominion",
    "Empire": "Dominion",
    "Republic": "Commonwealth",
    "Galactic": "Astral",
    "Star Wars": "Astral Frontiers",
    "BBY": "PSE",
    "ABY": "ASE",
}


class ParagraphExtractor(HTMLParser):
    """Extract article paragraphs while ignoring tables, citations, and media."""

    BLOCKED_TAGS = {"aside", "figure", "nav", "script", "style", "sup", "table"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocked_depth = 0
        self.in_paragraph = False
        self.current: list[str] = []
        self.paragraphs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.BLOCKED_TAGS:
            self.blocked_depth += 1
        elif tag == "p" and self.blocked_depth == 0:
            self.in_paragraph = True
            self.current = []
        elif tag == "br" and self.in_paragraph and self.blocked_depth == 0:
            self.current.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag == "p" and self.in_paragraph and self.blocked_depth == 0:
            self.paragraphs.append("".join(self.current))
            self.in_paragraph = False
            self.current = []
        if tag in self.BLOCKED_TAGS and self.blocked_depth:
            self.blocked_depth -= 1

    def handle_data(self, data: str) -> None:
        if self.in_paragraph and self.blocked_depth == 0:
            self.current.append(data)


BOILERPLATE_MARKERS = (
    "article has multiple issues",
    "article is in need of",
    "please help wookieepedia",
    "lucasfilm has not established",
    "conflicting sources for this article",
    "editor discretion is advised",
    "you may be looking for",
    "this is a disambiguation page",
    "see this article's talk page",
    "has been identified as containing",
)

CITATION_RE = re.compile(
    r"\[\s*(?:\d+|[a-z])(?:\s*[,;]\s*(?:\d+|[a-z]))*\s*\]",
    flags=re.IGNORECASE,
)
WORD_RE = re.compile(r"\b[\w'-]+\b", flags=re.UNICODE)


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def fetch_html(title: str, lead_only: bool = True) -> str:
    params = {
        "action": "parse",
        "page": title,
        "prop": "text",
        "redirects": "1",
        "format": "json",
        "formatversion": "2",
        "origin": "*",
    }
    if lead_only:
        params["section"] = "0"
    url = f"{API_URL}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                payload = json.load(response)
            if "error" in payload:
                raise RuntimeError(f"MediaWiki API error for {title}: {payload['error']}")
            return payload["parse"]["text"]
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            if attempt == 2:
                raise RuntimeError(f"Unable to download {title}: {error}") from error
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def clean_paragraphs(source_html: str) -> list[str]:
    parser = ParagraphExtractor()
    parser.feed(source_html)
    cleaned: list[str] = []

    for raw_paragraph in parser.paragraphs:
        paragraph = html.unescape(unicodedata.normalize("NFKC", raw_paragraph))
        paragraph = CITATION_RE.sub("", paragraph)
        paragraph = re.sub(r"https?://\S+", "", paragraph)
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        lowered = paragraph.casefold()
        if word_count(paragraph) < 25:
            continue
        if any(marker in lowered for marker in BOILERPLATE_MARKERS):
            continue
        cleaned.append(paragraph)
    return cleaned


def select_excerpt(paragraphs: list[str], title: str) -> list[str]:
    selected: list[str] = []
    total = 0

    for paragraph in paragraphs:
        count = word_count(paragraph)
        if total >= TARGET_WORDS:
            break
        if total + count <= MAX_WORDS:
            selected.append(paragraph)
            total += count
            continue

        room = MAX_WORDS - total
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        fitting: list[str] = []
        fitting_count = 0
        for sentence in sentences:
            sentence_words = word_count(sentence)
            if fitting_count + sentence_words > room:
                break
            fitting.append(sentence)
            fitting_count += sentence_words
        if fitting:
            selected.append(" ".join(fitting))
            total += fitting_count
        break

    if total < MIN_WORDS:
        raise ValueError(f"{title}: only {total} usable words after cleaning")
    return selected


def term_pattern(term: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", flags=re.IGNORECASE)


REPLACEMENTS = tuple(
    (term_pattern(source), target)
    for source, target in sorted(TERMS_MAP.items(), key=lambda item: len(item[0]), reverse=True)
)


def replace_terms(text: str) -> str:
    result = text
    for pattern, target in REPLACEMENTS:
        result = pattern.sub(target, result)
    return result


def build_document(entity: SourceEntity) -> tuple[str, int]:
    source_html = fetch_html(entity.source_title, lead_only=True)
    paragraphs = clean_paragraphs(source_html)

    if word_count(" ".join(paragraphs)) < MIN_WORDS:
        source_html = fetch_html(entity.source_title, lead_only=False)
        paragraphs = clean_paragraphs(source_html)

    excerpt = select_excerpt(paragraphs, entity.source_title)
    transformed = [replace_terms(paragraph) for paragraph in excerpt]
    body = "\n\n".join(transformed)
    document = f"# {entity.fictional_title}\n\nCategory: {entity.category}\n\n{body}\n"
    return document, word_count(body)


def validate_documents(word_counts: dict[str, int]) -> None:
    documents = sorted(OUTPUT_DIR.glob("*.md"))
    expected_names = {entity.filename for entity in ENTITIES}
    actual_names = {path.name for path in documents}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise ValueError(f"Unexpected document set; missing={missing}, extra={extra}")

    banned_markers = ("starwars", "wookieepedia", "lucasfilm", "fandom.com", "http://", "https://")
    source_patterns = tuple((source, term_pattern(source)) for source in TERMS_MAP)

    for path in documents:
        text = path.read_text(encoding="utf-8")
        if not text.startswith("# ") or "\n\nCategory: " not in text:
            raise ValueError(f"{path.name}: missing title or category")
        count = word_counts[path.name]
        if not MIN_WORDS <= count <= MAX_WORDS:
            raise ValueError(f"{path.name}: {count} words; expected {MIN_WORDS}-{MAX_WORDS}")
        lowered = text.casefold()
        leaked_markers = [marker for marker in banned_markers if marker in lowered]
        leaked_terms = [source for source, pattern in source_patterns if pattern.search(text)]
        if leaked_markers or leaked_terms:
            raise ValueError(
                f"{path.name}: canonical data remains; markers={leaked_markers}, "
                f"terms={leaked_terms[:10]}"
            )

    terms_path = OUTPUT_DIR / "terms_map.json"
    saved_map = json.loads(terms_path.read_text(encoding="utf-8"))
    if saved_map != TERMS_MAP:
        raise ValueError("terms_map.json differs from the mapping in the script")


def main() -> None:
    if len(ENTITIES) != 32:
        raise ValueError(f"Expected 32 source entities, got {len(ENTITIES)}")
    if len({entity.filename for entity in ENTITIES}) != len(ENTITIES):
        raise ValueError("Duplicate output filename")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    word_counts: dict[str, int] = {}

    for position, entity in enumerate(ENTITIES, start=1):
        document, count = build_document(entity)
        output_path = OUTPUT_DIR / entity.filename
        output_path.write_text(document, encoding="utf-8", newline="\n")
        word_counts[entity.filename] = count
        print(f"[{position:02d}/{len(ENTITIES)}] {entity.filename}: {count} words")

    terms_path = OUTPUT_DIR / "terms_map.json"
    terms_path.write_text(
        json.dumps(TERMS_MAP, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    validate_documents(word_counts)
    print(f"Validation passed: {len(ENTITIES)} documents and {len(TERMS_MAP)} replacements")


if __name__ == "__main__":
    main()
