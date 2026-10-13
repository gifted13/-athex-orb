"""Προεπιλεγμένο σύνολο μετοχών του Χρηματιστηρίου Αθηνών + προσθήκες χρήστη.

ΣΗΜΕΙΩΣΗ: τα σύμβολα είναι τα σύμβολα του Yahoo Finance (κατάληξη .AT).
Οι εταιρείες αλλάζουν ονόματα/σύμβολα (συγχωνεύσεις, μετονομασίες, διαγραφές),
οπότε αν κάποιο δεν επιστρέφει δεδομένα, απλώς παραλείπεται — και μπορείτε
να προσθέσετε οποιοδήποτε σύμβολο από το UI ή το data/universe_custom.json.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Stock:
    ticker: str            # π.χ. "OPAP.AT"
    name: str              # ελληνική ονομασία για εμφάνιση
    sector: str            # κλάδος στα ελληνικά
    aliases: list[str] = field(default_factory=list)  # για αναζήτηση (λατινικά, παλιά ονόματα κ.λπ.)
    custom: bool = False   # προστέθηκε από τον χρήστη

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_UNIVERSE: list[Stock] = [
    # Τράπεζες
    Stock("ETE.AT", "Εθνική Τράπεζα", "Τράπεζες", ["National Bank of Greece", "NBG", "ETE"]),
    Stock("EUROB.AT", "Eurobank", "Τράπεζες", ["Eurobank Ergasias", "Γιούρομπανκ"]),
    Stock("TPEIR.AT", "Τράπεζα Πειραιώς", "Τράπεζες", ["Piraeus Bank", "Peiraios"]),
    Stock("ALPHA.AT", "Alpha Bank", "Τράπεζες", ["Άλφα Τράπεζα", "Alpha Services"]),
    Stock("BOCHGR.AT", "Bank of Cyprus", "Τράπεζες", ["Τράπεζα Κύπρου"]),
    Stock("OPTIMA.AT", "Optima bank", "Τράπεζες", ["Οπτίμα"]),
    # Ενέργεια & κοινή ωφέλεια
    Stock("PPC.AT", "ΔΕΗ", "Ενέργεια", ["Public Power Corporation", "DEH", "PPC"]),
    Stock("MOH.AT", "Motor Oil", "Ενέργεια", ["Μότορ Όιλ", "Motor Oil Hellas"]),
    Stock("ELPE.AT", "HELLENiQ ENERGY", "Ενέργεια", ["Ελληνικά Πετρέλαια", "Hellenic Petroleum", "ΕΛΠΕ"]),
    Stock("ADMIE.AT", "ΑΔΜΗΕ Συμμετοχών", "Ενέργεια", ["ADMIE", "IPTO"]),
    # Βιομηχανία
    Stock("MTLN.AT", "Metlen", "Βιομηχανία", ["Μυτιληναίος", "Mytilineos", "MYTIL"]),
    Stock("CENER.AT", "Cenergy", "Βιομηχανία", ["Cenergy Holdings"]),
    Stock("VIO.AT", "Βιοχάλκο", "Βιομηχανία", ["Viohalco"]),
    Stock("ELHA.AT", "ΕΛΒΑΛΧΑΛΚΟΡ", "Βιομηχανία", ["ElvalHalcor", "Ελβάλ", "Χαλκόρ"]),
    Stock("TITC.AT", "Titan", "Βιομηχανία", ["Τιτάν", "Titan Cement"]),
    Stock("PLAKR.AT", "Πλαστικά Κρήτης", "Βιομηχανία", ["Plastika Kritis"]),
    # Κατασκευές & υποδομές
    Stock("GEKTERNA.AT", "ΓΕΚ ΤΕΡΝΑ", "Κατασκευές", ["GEK TERNA"]),
    Stock("AKTR.AT", "Aktor", "Κατασκευές", ["Άκτωρ", "Intrakat", "Ιντρακάτ"]),
    Stock("AIA.AT", "Διεθνής Αερολιμένας Αθηνών", "Υποδομές", ["Athens International Airport", "ΔΑΑ"]),
    Stock("PPA.AT", "ΟΛΠ", "Υποδομές", ["Piraeus Port Authority", "Λιμάνι Πειραιά"]),
    Stock("OLTH.AT", "ΟΛΘ", "Υποδομές", ["Thessaloniki Port Authority"]),
    Stock("EYDAP.AT", "ΕΥΔΑΠ", "Υποδομές", ["EYDAP", "Athens Water"]),
    Stock("EYAPS.AT", "ΕΥΑΘ", "Υποδομές", ["EYATH", "Thessaloniki Water"]),
    # Τηλεπικοινωνίες, ψυχαγωγία
    Stock("HTO.AT", "ΟΤΕ", "Τηλεπικοινωνίες", ["OTE", "Hellenic Telecom", "Cosmote"]),
    Stock("OPAP.AT", "ΟΠΑΠ (Allwyn)", "Ψυχαγωγία", ["OPAP", "Allwyn"]),
    # Λιανική, καταναλωτικά
    Stock("BELA.AT", "Jumbo", "Λιανική", ["Τζάμπο"]),
    Stock("FOYRK.AT", "Fourlis", "Λιανική", ["Φουρλής", "IKEA"]),
    Stock("SAR.AT", "Σαράντης", "Καταναλωτικά", ["Sarantis"]),
    Stock("KRI.AT", "Κρι Κρι", "Καταναλωτικά", ["Kri Kri"]),
    # Μεταφορές, ακίνητα
    Stock("AEGN.AT", "Aegean Airlines", "Μεταφορές", ["Αιγαίον", "Aegean"]),
    Stock("LAMDA.AT", "Lamda Development", "Ακίνητα", ["Λάμδα", "Ελληνικό"]),
]

TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9\-]{0,14}(\.AT)?$")


def normalize_ticker(raw: str) -> str | None:
    """Μετατρέπει είσοδο χρήστη σε σύμβολο Yahoo: 'opap' -> 'OPAP.AT'. Επιστρέφει None αν είναι άκυρο."""
    if not raw:
        return None
    t = raw.strip().upper().replace(" ", "")
    if not TICKER_RE.match(t):
        return None
    if not t.endswith(".AT"):
        t += ".AT"
    return t


def fold(text: str) -> str:
    """Κανονικοποίηση για αναζήτηση: πεζά, χωρίς τόνους/διαλυτικά, τελικό σ -> σ."""
    text = unicodedata.normalize("NFD", text or "")
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.lower().replace("ς", "σ").strip()


def custom_path(data_dir: Path) -> Path:
    return Path(data_dir) / "universe_custom.json"


def load_universe(data_dir: Path) -> list[Stock]:
    """Προεπιλεγμένες μετοχές + όσες έχει προσθέσει ο χρήστης."""
    stocks = {s.ticker: Stock(**s.to_dict()) for s in DEFAULT_UNIVERSE}
    p = custom_path(data_dir)
    if p.exists():
        try:
            for item in json.loads(p.read_text(encoding="utf-8")):
                s = Stock(ticker=item["ticker"], name=item.get("name") or item["ticker"].removesuffix(".AT"),
                          sector=item.get("sector") or "Άλλο", aliases=item.get("aliases", []), custom=True)
                stocks.setdefault(s.ticker, s)
        except (ValueError, KeyError, TypeError):
            pass
    return list(stocks.values())


def add_custom(data_dir: Path, stock: Stock) -> None:
    p = custom_path(data_dir)
    items = []
    if p.exists():
        try:
            items = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            items = []
    items = [i for i in items if i.get("ticker") != stock.ticker]
    stock.custom = True
    items.append(stock.to_dict())
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def remove_custom(data_dir: Path, ticker: str) -> bool:
    p = custom_path(data_dir)
    if not p.exists():
        return False
    items = json.loads(p.read_text(encoding="utf-8"))
    kept = [i for i in items if i.get("ticker") != ticker]
    p.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(kept) != len(items)


def search(stocks: list[Stock], query: str, limit: int = 20) -> list[Stock]:
    """Αναζήτηση χωρίς τόνους σε σύμβολο, όνομα, κλάδο και ψευδώνυμα. Πρώτα τα καλύτερα ταιριάσματα."""
    q = fold(query)
    if len(q) > 3 and q.endswith(".at"):   # πλήρες σύμβολο Yahoo, π.χ. «OPAP.AT»
        q = q[:-3]
    if not q:
        return stocks[:limit]
    scored = []
    for s in stocks:
        tick = fold(s.ticker.removesuffix(".AT"))
        fields_ = [fold(s.name)] + [fold(a) for a in s.aliases]
        if tick == q:
            score = 0
        elif tick.startswith(q):
            score = 1
        elif any(f.startswith(q) for f in fields_):
            score = 2
        elif any(q in f for f in fields_) or q in tick:
            score = 3
        elif q in fold(s.sector):
            score = 4
        else:
            continue
        scored.append((score, s.ticker, s))
    scored.sort(key=lambda x: (x[0], x[1]))
    return [s for _, _, s in scored[:limit]]
