"""«Η Σφαίρα»: μοντέλο τριών καταστάσεων (στόχος πρώτα / stop πρώτα / λήξη χρόνου) και walk‑forward.

* Προεπιλογή: πολυωνυμική λογιστική παλινδρόμηση με τυποποίηση (σταθερή και εξηγήσιμη).
  Εναλλακτικά "hgb" (HistGradientBoosting), χωρίς ανάλυση συνεισφορών.
* Walk‑forward χωρίς «ματιά στο μέλλον»: στην ημερομηνία επανεκπαίδευσης T το μοντέλο βλέπει μόνο
  δείγματα των οποίων το αποτέλεσμα κρίθηκε αυστηρά πριν από την T (`resolved_date < T`).
* Αν από το σύνολο εκπαίδευσης λείπει κάποια κατάσταση, παίρνει μικρή πιθανότητα‑«βάση»
  (ABSENT_CLASS_PRIOR) και οι υπόλοιπες κανονικοποιούνται ώστε το άθροισμα να είναι 1.
"""
from __future__ import annotations

import contextlib

import numpy as np
import pandas as pd

# Εισαγωγές του scikit-learn στο επίπεδο του module (όχι «τεμπέλικα» μέσα σε συναρτήσεις): η πρώτη εισαγωγή
# ενός πακέτου από δύο νήματα ταυτόχρονα (εργασία εκπαίδευσης + αίτημα που ξεφορτώνει το model.pkl)
# προκαλεί _DeadlockError στο σύστημα εισαγωγών της Python.
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # pragma: no cover
    threadpool_limits = None

from .features import FEATURE_COLUMNS
from .labels import LABEL_SL, LABEL_TO, LABEL_TP

N_CLASSES = 3
CLASSES = np.array([LABEL_TP, LABEL_SL, LABEL_TO])
ABSENT_CLASS_PRIOR = 0.01       # πιθανότητα για κατάσταση που δεν εμφανίστηκε ποτέ στην εκπαίδευση
MIN_TRAIN_SAMPLES = 200         # ελάχιστα επιλυμένα δείγματα για εκπαίδευση
Z_CLIP = 5.0                    # ακραίες τιμές (μετά την τυποποίηση) περικόπτονται στο ±5
LOGREG_C = 0.5
LOGREG_TOL = 1e-4               # ανοχή σύγκλισης (με «ζεστή» εκκίνηση η λύση διαφέρει < 0,1% σε πιθανότητα)
MODEL_KINDS = ("logreg", "hgb")
PRED_COLUMNS = ["p_tp", "p_sl", "p_to", "timeout_mean", "model_date"]


def _single_thread():
    """Ένα νήμα για BLAS/OpenMP: για πίνακες αυτού του μεγέθους τα πολλά νήματα απλώς καθυστερούν."""
    if threadpool_limits is None:  # pragma: no cover
        return contextlib.nullcontext()
    return threadpool_limits(limits=1)


def clip_z(X):
    """Περικοπή τυποποιημένων τιμών στο [−Z_CLIP, Z_CLIP] (προστασία από λανθασμένα/ακραία δεδομένα)."""
    return np.clip(X, -Z_CLIP, Z_CLIP)


def _make_estimator(kind: str, random_state: int):
    if kind == "logreg":
        return Pipeline([
            ("scale", StandardScaler()),
            ("clip", FunctionTransformer(clip_z)),
            ("clf", LogisticRegression(C=LOGREG_C, class_weight=None, max_iter=2000, tol=LOGREG_TOL,
                                       random_state=random_state)),
        ])
    if kind == "hgb":
        return Pipeline([
            ("scale", StandardScaler()),
            ("clip", FunctionTransformer(clip_z)),
            ("clf", HistGradientBoostingClassifier(learning_rate=0.08, max_iter=120, max_leaf_nodes=15,
                                                   min_samples_leaf=50, l2_regularization=1.0,
                                                   early_stopping=False, random_state=random_state)),
        ])
    raise ValueError(f"Άγνωστος τύπος μοντέλου «{kind}» (επιτρέπονται: {', '.join(MODEL_KINDS)})")


class OrbModel:
    """Μοντέλο πιθανοτήτων [p_tp, p_sl, p_to] για μια θέση που ανοίγει στο επόμενο άνοιγμα."""

    def __init__(self, kind: str = "logreg", random_state: int = 0, feature_names: list[str] | None = None):
        if kind not in MODEL_KINDS:
            raise ValueError(f"Άγνωστος τύπος μοντέλου «{kind}» (επιτρέπονται: {', '.join(MODEL_KINDS)})")
        self.kind = kind
        self.random_state = random_state
        self.feature_names = list(feature_names) if feature_names is not None else list(FEATURE_COLUMNS)
        self.class_timeout_mean: float = 0.0
        self.trained_until: pd.Timestamp | None = None
        self.n_train: int = 0
        self.class_counts: dict[int, int] = {int(c): 0 for c in CLASSES}
        self._est = None                 # εκπαιδευμένο Pipeline (αν υπάρχουν ≥ 2 καταστάσεις)
        self._present = np.array([], dtype="int64")    # καταστάσεις που εμφανίστηκαν, με τη σειρά του estimator
        self._const: np.ndarray | None = None          # σταθερές πιθανότητες αν υπάρχει μόνο μία κατάσταση

    # ------------------------------------------------------------ εκπαίδευση

    def fit(self, X: np.ndarray, y: np.ndarray, gross_ret: np.ndarray | None = None,
            trained_until=None, warm_start_from: "OrbModel | None" = None) -> "OrbModel":
        """Εκπαίδευση.

        * `gross_ret` (προαιρετικό): μικτή απόδοση κάθε δείγματος, για το class_timeout_mean.
        * `trained_until`: η ημερομηνία «γνώσης» του μοντέλου (μόνο για πληροφόρηση).
        * `warm_start_from`: προηγούμενο μοντέλο logreg ως αρχικό σημείο της βελτιστοποίησης
          (ίδια λύση — το πρόβλημα είναι κυρτό — απλώς ταχύτερα· χρησιμοποιείται στο walk‑forward).
        """
        X = np.asarray(X, dtype="float64")
        y_raw = np.asarray(y, dtype="float64")
        if X.ndim != 2 or len(X) != len(y_raw):
            raise ValueError("Τα X και y πρέπει να έχουν τον ίδιο αριθμό γραμμών")
        if len(y_raw) == 0:
            raise ValueError("Δεν υπάρχουν δείγματα για εκπαίδευση")
        if not np.isfinite(X).all():
            raise ValueError("Τα δεδομένα εκπαίδευσης περιέχουν κενές (NaN) ή άπειρες τιμές")
        if not (np.isfinite(y_raw).all() and np.isin(y_raw, CLASSES).all()):
            raise ValueError("Άγνωστη ή κενή ετικέτα στα δεδομένα εκπαίδευσης")
        y = y_raw.astype("int64")
        if X.shape[1] != len(self.feature_names):
            self.feature_names = [f"x{i}" for i in range(X.shape[1])]

        present = np.unique(y)
        self.class_counts = {int(c): int((y == c).sum()) for c in CLASSES}
        self.n_train = int(len(y))
        self.trained_until = pd.Timestamp(trained_until) if trained_until is not None else None

        self.class_timeout_mean = 0.0
        if gross_ret is not None:
            g = np.asarray(gross_ret, dtype="float64")
            sel = (y == LABEL_TO) & np.isfinite(g)
            if sel.any():
                self.class_timeout_mean = float(g[sel].mean())

        self._const = None
        self._est = None
        if len(present) == 1:
            self._present = present
            const = np.full(N_CLASSES, ABSENT_CLASS_PRIOR)
            const[int(present[0])] = 1.0 - ABSENT_CLASS_PRIOR * (N_CLASSES - 1)
            self._const = const
            return self

        est = _make_estimator(self.kind, self.random_state)
        if (self.kind == "logreg" and warm_start_from is not None and warm_start_from._est is not None
                and warm_start_from.kind == "logreg" and np.array_equal(warm_start_from._present, present)
                and warm_start_from._est[-1].coef_.shape[1] == X.shape[1]):
            prev = warm_start_from._est[-1]
            clf = est[-1]
            clf.set_params(warm_start=True)
            clf.coef_ = prev.coef_.copy()
            clf.intercept_ = prev.intercept_.copy()
        with self._threads():
            est.fit(X, y)
        self._est = est
        self._present = np.asarray(est.classes_, dtype="int64")
        return self

    def _threads(self):
        """Η logreg τρέχει ταχύτερα σε ένα νήμα· το hgb ωφελείται από πολλά."""
        return _single_thread() if self.kind == "logreg" else contextlib.nullcontext()

    @property
    def is_fitted(self) -> bool:
        return self._est is not None or self._const is not None

    # ------------------------------------------------------------ πρόβλεψη

    def predict_proba(self, X) -> np.ndarray:
        """Πίνακας (n, 3) με στήλες [p_tp, p_sl, p_to]· κάθε γραμμή αθροίζει σε 1.
        Γραμμές με κενές τιμές (NaN) επιστρέφουν NaN."""
        if not self.is_fitted:
            raise RuntimeError("Το μοντέλο δεν έχει εκπαιδευτεί")
        X = np.asarray(X, dtype="float64")
        if X.ndim == 1:
            X = X[None, :]
        n = len(X)
        out = np.full((n, N_CLASSES), np.nan)
        ok = np.isfinite(X).all(axis=1)
        if not ok.any():
            return out
        if self._const is not None:
            out[ok] = self._const
            return out

        with self._threads():
            raw = self._est.predict_proba(X[ok])
        n_absent = N_CLASSES - len(self._present)
        scale = 1.0 - ABSENT_CLASS_PRIOR * n_absent
        probs = np.full((int(ok.sum()), N_CLASSES), ABSENT_CLASS_PRIOR if n_absent else 0.0)
        probs[:, self._present] = raw * scale
        probs /= probs.sum(axis=1, keepdims=True)
        out[ok] = probs
        return out

    def contributions(self, x_row) -> dict[str, float]:
        """Συνεισφορά κάθε χαρακτηριστικού στο log(p_tp / p_sl) = logit(p_tp) − logit(p_sl) (σχετικά).

        logreg: (συντελεστής TP − συντελεστής SL) × τυποποιημένη τιμή. hgb ή χωρίς TP/SL στην εκπαίδευση: {}.
        """
        if self.kind != "logreg" or self._est is None:
            return {}
        if LABEL_TP not in self._present or LABEL_SL not in self._present:
            return {}
        x = np.asarray(x_row, dtype="float64").reshape(1, -1)
        if x.shape[1] != len(self.feature_names) or not np.isfinite(x).all():
            return {}
        z = self._est[:-1].transform(x)[0]
        diff = self._coef_by_class()
        d = diff[LABEL_TP] - diff[LABEL_SL]
        return {name: float(v) for name, v in zip(self.feature_names, d * z)}

    def base_log_ratio(self) -> float | None:
        """Ο σταθερός όρος του log(p_tp / p_sl): η «βάση» για μια μέση περίπτωση (τυποποιημένα
        χαρακτηριστικά = 0), πριν από τις συνεισφορές των παραγόντων. Αντανακλά τις ιστορικές συχνότητες."""
        if self.kind != "logreg" or self._est is None:
            return None
        if LABEL_TP not in self._present or LABEL_SL not in self._present:
            return None
        clf = self._est[-1]
        b = np.asarray(clf.intercept_, dtype="float64")
        classes = [int(c) for c in clf.classes_]
        if b.shape[0] == 1 and len(classes) == 2:
            by = {classes[0]: 0.0, classes[1]: float(b[0])}
        else:
            by = {c: float(b[i]) for i, c in enumerate(classes)}
        return float(by[LABEL_TP] - by[LABEL_SL])

    def _coef_by_class(self) -> dict[int, np.ndarray]:
        """Συντελεστές ανά κατάσταση (για 2 καταστάσεις: η πρώτη έχει μηδενικούς, όπως στο sklearn)."""
        clf = self._est[-1]
        coef = np.asarray(clf.coef_, dtype="float64")
        classes = [int(c) for c in clf.classes_]
        if coef.shape[0] == 1 and len(classes) == 2:
            return {classes[0]: np.zeros(coef.shape[1]), classes[1]: coef[0]}
        return {c: coef[i] for i, c in enumerate(classes)}

    def info(self) -> dict:
        """Μεταδεδομένα για την κατάσταση της εφαρμογής."""
        return {
            "kind": self.kind,
            "n_train": self.n_train,
            "trained_until": self.trained_until.strftime("%Y-%m-%d") if self.trained_until is not None else None,
            "class_counts": {"tp": self.class_counts.get(LABEL_TP, 0), "sl": self.class_counts.get(LABEL_SL, 0),
                             "to": self.class_counts.get(LABEL_TO, 0)},
            "class_timeout_mean": self.class_timeout_mean,
        }


# ---------------------------------------------------------------- walk‑forward

def _report(progress, fraction: float, message_el: str) -> None:
    if progress is not None:
        try:
            progress(float(min(max(fraction, 0.0), 1.0)), message_el)
        except Exception:  # η ένδειξη προόδου δεν πρέπει να σταματά την εκπαίδευση
            pass


def _empty_preds(index: pd.Index | None = None) -> pd.DataFrame:
    if index is None:
        index = pd.MultiIndex.from_arrays([pd.DatetimeIndex([]).as_unit("us"), pd.Index([], dtype="str")],
                                          names=["date", "ticker"])
    df = pd.DataFrame({c: pd.Series(np.full(len(index), np.nan), index=index, dtype="float64")
                       for c in PRED_COLUMNS[:-1]}, index=index)
    df["model_date"] = pd.Series(pd.NaT, index=index, dtype="datetime64[us]")
    return df


def prepare_training(panel: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """Πίνακες numpy για το walk‑forward (κοινοί και για ελέγχους/tests).

    `avail_code[i]` = ο δείκτης της πρώτης ημερομηνίας του panel στην οποία το δείγμα i είναι
    διαθέσιμο για εκπαίδευση, δηλαδή η πρώτη ημερομηνία T με resolved_date < T (= πλήθος ημερομηνιών
    για ό,τι κρίθηκε την τελευταία ημέρα — διαθέσιμο μόνο στο τελικό μοντέλο).
    """
    if not labels.index.equals(panel.index):
        labels = labels.reindex(panel.index)
    dates = panel.index.get_level_values("date")
    uniq = pd.DatetimeIndex(dates.unique()).sort_values()
    date_code = uniq.get_indexer(dates)

    X = panel[FEATURE_COLUMNS].to_numpy(dtype="float64", na_value=np.nan)
    valid = np.isfinite(X).all(axis=1)
    y = labels["label"].to_numpy(dtype="float64", na_value=np.nan)
    resolved = pd.DatetimeIndex(labels["resolved_date"])
    has_res = ~resolved.isna()
    usable = valid & np.isfinite(y) & np.asarray(has_res)

    avail_code = np.full(len(panel), np.iinfo("int64").max, dtype="int64")
    if usable.any():
        # searchsorted «right»: πλήθος ημερομηνιών ≤ resolved_date ⇒ πρώτη ημερομηνία με resolved_date < T.
        # Αποτελέσματα που κρίνονται μετά την τελευταία ημερομηνία του panel δεν είναι ποτέ διαθέσιμα.
        r = resolved[usable]
        code = uniq.searchsorted(r, side="right").astype("int64")
        code[np.asarray(r > uniq[-1])] = np.iinfo("int64").max
        avail_code[usable] = code
    gross = labels["gross_ret"].to_numpy(dtype="float64", na_value=np.nan)
    return {"uniq": uniq, "date_code": date_code, "X": X, "valid": valid, "y": y, "usable": usable,
            "avail_code": avail_code, "gross": gross}


def _train_mask(d: dict, i: int, window: int) -> np.ndarray:
    m = d["usable"] & (d["avail_code"] <= i)
    if window > 0:
        m &= d["date_code"] >= i - window
    return m


def walk_forward(panel: pd.DataFrame, labels: pd.DataFrame, cfg, progress=None
                 ) -> tuple[pd.DataFrame, OrbModel | None]:
    """Εκπαίδευση/πρόβλεψη κυλιόμενα στον χρόνο.

    Επιστρέφει (preds, final_model): `preds` με στήλες p_tp, p_sl, p_to, timeout_mean, model_date για τις
    γραμμές του panel με πλήρη χαρακτηριστικά από την πρώτη ημερομηνία πρόβλεψης και μετά, και το τελικό
    μοντέλο εκπαιδευμένο σε όλα τα επιλυμένα δείγματα (για τα ζωντανά σήματα).
    """
    mc = cfg.model
    kind = mc.kind if mc.kind in MODEL_KINDS else "logreg"
    min_days = max(int(mc.min_train_days), 1)
    every = max(int(mc.retrain_every_days), 1)
    window = max(int(mc.train_window_days), 0)

    if panel is None or not len(panel):
        return _empty_preds(), None
    d = prepare_training(panel, labels)
    uniq, date_code, X = d["uniq"], d["date_code"], d["X"]
    n_dates = len(uniq)

    def fit_at(i: int, until, prev: OrbModel | None) -> OrbModel | None:
        m = _train_mask(d, i, window)
        if int(m.sum()) < MIN_TRAIN_SAMPLES:
            return None
        model = OrbModel(kind=kind, random_state=0)
        return model.fit(X[m], d["y"][m], gross_ret=d["gross"][m], trained_until=until, warm_start_from=prev)

    # πρώτη ημερομηνία πρόβλεψης: ≥ min_train_days ημερομηνίες ιστορικού και ≥ MIN_TRAIN_SAMPLES δείγματα
    first = None
    for i in range(min_days, n_dates):
        if int(_train_mask(d, i, window).sum()) >= MIN_TRAIN_SAMPLES:
            first = i
            break

    pieces_idx, pieces_p, pieces_tm, pieces_md = [], [], [], []
    model: OrbModel | None = None
    if first is not None:
        starts = list(range(first, n_dates, every))
        for k, i in enumerate(starts):
            T = uniq[i]
            _report(progress, k / max(len(starts), 1),
                    f"Εκπαίδευση μοντέλου {k + 1}/{len(starts)} (έως {T:%d/%m/%Y})")
            new = fit_at(i, T, model)
            if new is not None:
                model = new
            if model is None:
                continue
            end = min(i + every, n_dates)
            rows = np.flatnonzero((date_code >= i) & (date_code < end) & d["valid"])
            if not len(rows):
                continue
            pieces_idx.append(rows)
            pieces_p.append(model.predict_proba(X[rows]))
            pieces_tm.append(np.full(len(rows), model.class_timeout_mean))
            pieces_md.append(np.full(len(rows), model.trained_until.to_datetime64(), dtype="datetime64[us]"))

    if pieces_idx:
        rows = np.concatenate(pieces_idx)
        p = np.vstack(pieces_p)
        preds = pd.DataFrame({
            "p_tp": p[:, 0], "p_sl": p[:, 1], "p_to": p[:, 2],
            "timeout_mean": np.concatenate(pieces_tm),
            "model_date": np.concatenate(pieces_md).astype("datetime64[us]"),
        }, index=panel.index[rows])
        preds = preds.sort_index()
    else:
        preds = _empty_preds()

    # τελικό μοντέλο: όλα τα επιλυμένα δείγματα μέχρι την τελευταία ημερομηνία (για ζωντανά σήματα)
    final = None
    if n_dates >= min_days:
        final = fit_at(n_dates, uniq[-1], model)
    _report(progress, 1.0, "Η εκπαίδευση του μοντέλου ολοκληρώθηκε")
    return preds, final
