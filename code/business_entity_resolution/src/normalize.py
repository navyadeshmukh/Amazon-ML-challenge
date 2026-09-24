"""Stage 1 - text normalisation.

Original columns are kept; we ADD normalised columns so nothing is destroyed.
Country is treated as an open set of strings (France is unseen in training).
"""
import re
import unicodedata
import pandas as pd

LEGAL_SUFFIXES = {
    "pvt", "private", "ltd", "limited", "llp", "llc", "inc", "incorporated",
    "corp", "corporation", "co", "company", "plc", "lp", "opc", "gmbh",
    "sarl", "sas", "sa", "eurl", "sasu", "snc", "ets", "etablissements",
}
NAME_STOP = {"and", "the", "of", "m/s", "ms"}
NAME_ABBR = {"intl": "international", "natl": "national", "assoc": "associates",
             "svc": "service", "svcs": "services", "mfg": "manufacturing",
             "dept": "department", "univ": "university", "bros": "brothers",
             "ent": "enterprises", "ents": "enterprises", "traders": "traders"}

ADDR_ABBR = {
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "bd": "boulevard", "ln": "lane", "dr": "drive",
    "hwy": "highway", "nr": "near", "opp": "opposite", "bldg": "building",
    "flr": "floor", "sec": "sector", "apt": "apartment", "ste": "suite",
    "rte": "route", "pl": "place", "sq": "square", "ch": "chemin",
    "cross": "cross", "extn": "extension", "ext": "extension", "colony": "colony",
    "nagar": "nagar", "mkt": "market", "cir": "circle", "ct": "court",
    "n": "north", "s": "south", "e": "east", "w": "west",
}
LANDMARK_RE = re.compile(r"\b(near|opposite|behind|beside|next to|adjacent|opp)\b.*$")
PIN_RE = re.compile(r"\b(\d{6}|\d{5})(?:-\d{4})?\b")
COUNTRY_ALIAS = {"usa": "us", "united states": "us", "united states of america": "us",
                 "u.s.": "us", "u.s.a.": "us", "america": "us",
                 "in": "india", "ind": "india", "bharat": "india",
                 "fr": "france", "république française": "france"}


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def _base(s) -> str:
    s = "" if s is None or (isinstance(s, float) and pd.isna(s)) else str(s)
    s = strip_accents(s).lower().replace("&", " and ").replace("'", "").replace("`", "")
    s = re.sub(r"\b(\w)\.", r"\1", s)          # m.g. -> mg
    s = re.sub(r"[^a-z0-9/ ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def norm_name(s) -> str:
    toks = [NAME_ABBR.get(t, t) for t in _base(s).split()]
    toks = [t for t in toks if t not in LEGAL_SUFFIXES and t not in NAME_STOP]
    return " ".join(toks)


def norm_address(s) -> str:
    toks = [ADDR_ABBR.get(t, t) for t in _base(s).split()]
    return " ".join(toks)


def address_no_landmark(a: str) -> str:
    return LANDMARK_RE.sub("", a).strip()


def extract_pin(raw) -> str:
    m = PIN_RE.search("" if pd.isna(raw) else str(raw))
    return m.group(1)[:6] if m else ""


def norm_country(s) -> str:
    c = _base(s)
    return COUNTRY_ALIAS.get(c, c)


def add_normalized(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ("business_name", "business_address", "country"):
        if c not in df:
            df[c] = ""
        df[c] = df[c].fillna("").astype(str)
    df["name_n"] = df["business_name"].map(norm_name)
    df["addr_n"] = df["business_address"].map(norm_address)
    df["addr_core"] = df["addr_n"].map(address_no_landmark)
    df["pin"] = df["business_address"].map(extract_pin)
    df["country_n"] = df["country"].map(norm_country)
    df["name_compact"] = df["name_n"].str.replace(" ", "", regex=False)
    df["full_n"] = (df["name_n"] + " " + df["addr_core"]).str.strip()
    return df
