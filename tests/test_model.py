"""«Η Σφαίρα»: μοντέλο τριών καταστάσεων και walk‑forward χωρίς «ματιά στο μέλλον»."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from athex_orb import features, labels, orb_model
from athex_orb.features import FEATURE_COLUMNS
from athex_orb.labels import LABEL_SL, LABEL_TO, LABEL_TP
from athex_orb.orb_model import ABSENT_CLASS_PRIOR, OrbModel, walk_forward


def wf_cfg(session_cfg, min_train_days=300, every=21, window=400):
    c = copy.deepcopy(session_cfg)
    c.model.min_train_days = min_train_days
    c.model.retrain_every_days = every
    c.model.train_window_days = window
    return c


def toy_data(n=600, seed=0, classes=(0, 1, 2)):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, len(FEATURE_COLUMNS)))
    logits = np.column_stack([X[:, 0], -X[:, 0], 0.3 * X[:, 1]])[:, list(classes)]
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y = np.array([classes[rng.choice(len(classes), p=row)] for row in p])
    return X, y


# ---------------------------------------------------------------- 1. καμία «ματιά στο μέλλον»

@pytest.fixture
def fit_spy(monkeypatch):
    """Καταγράφει κάθε εκπαίδευση: (trained_until, X, y)."""
    calls = []
    original = OrbModel.fit

    def spy(self, X, y, *args, **kwargs):
        calls.append((kwargs.get("trained_until"), np.array(X, copy=True), np.array(y, copy=True)))
        return original(self, X, y, *args, **kwargs)

    monkeypatch.setattr(OrbModel, "fit", spy)
    return calls


@pytest.mark.parametrize("window", [400, 0])
def test_walk_forward_never_trains_on_unresolved_outcomes(panel, labels_df, session_cfg, fit_spy, window):
    cfg = wf_cfg(session_cfg, window=window)
    preds, final = walk_forward(panel, labels_df, cfg)
    assert len(preds) > 0 and final is not None
    assert len(fit_spy) >= 3

    # αντιστοίχιση γραμμής χαρακτηριστικών → (ημερομηνία, resolved_date) για τις γραμμές του panel
    X = panel[FEATURE_COLUMNS].to_numpy(dtype="float64")
    ok = np.isfinite(X).all(axis=1)
    resolved = pd.DatetimeIndex(labels_df["resolved_date"])
    dates = pd.DatetimeIndex(panel.index.get_level_values("date"))
    lookup = {X[i].tobytes(): (dates[i], resolved[i]) for i in np.flatnonzero(ok)}
    uniq = pd.DatetimeIndex(dates.unique()).sort_values()
    last = uniq[-1]

    model_dates = set(pd.DatetimeIndex(preds["model_date"]).unique())
    *wf_calls, final_call = fit_spy
    assert final_call[0] == last
    for T, Xt, yt in wf_calls:
        T = pd.Timestamp(T)
        info = [lookup[row.tobytes()] for row in Xt]        # KeyError ⇒ εκπαίδευση σε άγνωστη γραμμή
        d_train = pd.DatetimeIndex([a for a, _ in info])
        r_train = pd.DatetimeIndex([b for _, b in info])
        assert not r_train.isna().any(), "εκπαίδευση σε ανεπίλυτο δείγμα"
        assert r_train.max() < T, f"δείγμα με resolved_date {r_train.max()} ≥ ημερομηνία εκπαίδευσης {T}"
        assert d_train.max() < T
        if window:
            i = uniq.get_loc(T)
            assert d_train.min() >= uniq[max(i - window, 0)]
        assert T in model_dates
        assert len(Xt) >= orb_model.MIN_TRAIN_SAMPLES

    # το τελικό μοντέλο: όλα τα επιλυμένα έως την τελευταία ημερομηνία
    r_final = pd.DatetimeIndex([lookup[row.tobytes()][1] for row in final_call[1]])
    assert r_final.max() <= last
    usable = ok & np.isfinite(labels_df["label"].to_numpy(dtype="float64", na_value=np.nan))
    if window:
        usable &= np.asarray(dates >= uniq[len(uniq) - window])
    assert final.n_train == int(usable.sum())

    # κάθε πρόβλεψη χρησιμοποιεί μοντέλο εκπαιδευμένο έως (και όχι μετά) την ημερομηνία της
    pdates = pd.DatetimeIndex(preds.index.get_level_values("date"))
    assert (pd.DatetimeIndex(preds["model_date"]) <= pdates).all()


def test_outcomes_resolving_later_are_never_used(panel, labels_df, session_cfg):
    """Αν όλα τα αποτελέσματα «κρίνονται» μετά το τέλος των δεδομένων, δεν υπάρχει τίποτα για εκπαίδευση."""
    cfg = wf_cfg(session_cfg)
    late = labels_df.copy()
    late["resolved_date"] = late["resolved_date"] + pd.Timedelta(days=4000)
    preds, final = walk_forward(panel, late, cfg)
    assert len(preds) == 0
    assert final is None
    assert list(preds.columns) == orb_model.PRED_COLUMNS


def test_walk_forward_shift_of_resolution_delays_first_prediction(panel, labels_df, session_cfg):
    cfg = wf_cfg(session_cfg)
    p0, _ = walk_forward(panel, labels_df, cfg)
    late = labels_df.copy()
    late["resolved_date"] = late["resolved_date"] + pd.Timedelta(days=120)
    p1, _ = walk_forward(panel, late, cfg)
    first0 = p0.index.get_level_values("date").min()
    first1 = p1.index.get_level_values("date").min()
    assert first1 > first0


def test_walk_forward_structure(panel, labels_df, session_cfg):
    cfg = wf_cfg(session_cfg, min_train_days=300, every=21)
    seen = []
    preds, final = walk_forward(panel, labels_df, cfg, progress=lambda f, m: seen.append((f, m)))
    assert list(preds.columns) == orb_model.PRED_COLUMNS
    assert preds.index.isin(panel.index).all()
    assert preds.index.is_monotonic_increasing and preds.index.is_unique
    # μόνο γραμμές με πλήρη χαρακτηριστικά
    assert np.isfinite(panel.loc[preds.index, FEATURE_COLUMNS].to_numpy(dtype="float64")).all()
    uniq = pd.DatetimeIndex(panel.index.get_level_values("date").unique()).sort_values()
    first = preds.index.get_level_values("date").min()
    assert uniq.get_loc(first) >= 300
    # επανεκπαίδευση κάθε 21 ημερομηνίες
    md = pd.DatetimeIndex(sorted(set(preds["model_date"])))
    pos = np.array([uniq.get_loc(d) for d in md])
    assert (np.diff(pos) == 21).all()
    assert md[0] == first
    # μετά την πρώτη ημερομηνία πρόβλεψης όλες οι γραμμές με πλήρη χαρακτηριστικά έχουν πρόβλεψη
    later = panel[panel.index.get_level_values("date") >= first]
    later_ok = np.isfinite(later[FEATURE_COLUMNS].to_numpy(dtype="float64")).all(axis=1)
    assert len(preds) == int(later_ok.sum())
    p = preds[["p_tp", "p_sl", "p_to"]].to_numpy()
    assert np.allclose(p.sum(axis=1), 1.0) and (p > 0).all()
    assert np.isfinite(preds["timeout_mean"]).all()
    assert final.trained_until == uniq[-1]
    assert seen and seen[-1][0] == 1.0 and all(isinstance(m, str) and m for _, m in seen)


def test_walk_forward_not_enough_history(panel, labels_df, session_cfg):
    cfg = wf_cfg(session_cfg, min_train_days=5000)
    preds, final = walk_forward(panel, labels_df, cfg)
    assert len(preds) == 0 and final is None


# ---------------------------------------------------------------- 2. πάντα 3 πιθανότητες

@pytest.mark.parametrize("kind", ["logreg", "hgb"])
def test_predict_proba_three_columns_sum_to_one(kind):
    X, y = toy_data()
    m = OrbModel(kind=kind).fit(X, y)
    p = m.predict_proba(X[:50])
    assert p.shape == (50, 3)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert (p >= 0).all()
    assert m.predict_proba(X[0]).shape == (1, 3)         # και για μία γραμμή


@pytest.mark.parametrize("kind", ["logreg", "hgb"])
@pytest.mark.parametrize("present,absent", [((0, 1), 2), ((1, 2), 0), ((0, 2), 1)])
def test_missing_class_gets_small_prior(kind, present, absent):
    X, y = toy_data(classes=present)
    assert absent not in set(y)
    m = OrbModel(kind=kind).fit(X, y)
    p = m.predict_proba(X[:100])
    assert p.shape == (100, 3)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.allclose(p[:, absent], ABSENT_CLASS_PRIOR)
    assert (p[:, list(present)] > 0).all()


def test_single_class_training_set():
    X, _ = toy_data(n=300)
    y = np.full(300, LABEL_SL)
    m = OrbModel().fit(X, y)
    p = m.predict_proba(X[:10])
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.allclose(p[:, 1], 1 - 2 * ABSENT_CLASS_PRIOR)
    assert m.contributions(X[0]) == {}


def test_nan_rows_predict_nan_and_unfitted_raises():
    X, y = toy_data()
    m = OrbModel().fit(X, y)
    Xn = X[:5].copy()
    Xn[2, 3] = np.nan
    p = m.predict_proba(Xn)
    assert np.isnan(p[2]).all() and np.allclose(p[[0, 1, 3, 4]].sum(axis=1), 1.0)
    with pytest.raises(RuntimeError):
        OrbModel().predict_proba(X[:2])
    with pytest.raises(ValueError):
        OrbModel(kind="svm")
    with pytest.raises(ValueError):
        OrbModel().fit(X[:10], np.array([0, 1, 2, 3, 0, 1, 2, 0, 1, 2]))
    with pytest.raises(ValueError):
        OrbModel().fit(np.full((10, len(FEATURE_COLUMNS)), np.nan), np.zeros(10))


def test_walk_forward_handles_missing_timeout_class(panel, labels_df, session_cfg):
    cfg = wf_cfg(session_cfg)
    lab = labels_df.copy()
    lab.loc[lab["label"] == LABEL_TO, "label"] = LABEL_SL      # καμία «λήξη χρόνου» στην εκπαίδευση
    preds, final = walk_forward(panel, lab, cfg)
    p = preds[["p_tp", "p_sl", "p_to"]].to_numpy()
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.allclose(p[:, 2], ABSENT_CLASS_PRIOR)
    assert final.class_timeout_mean == 0.0


def test_fit_metadata():
    X, y = toy_data()
    g = np.where(y == LABEL_TO, 0.004, 0.05)
    m = OrbModel().fit(X, y, gross_ret=g, trained_until="2022-03-04")
    assert m.n_train == len(y)
    assert m.class_timeout_mean == pytest.approx(0.004)
    assert m.trained_until == pd.Timestamp("2022-03-04")
    info = m.info()
    assert info["trained_until"] == "2022-03-04"
    assert sum(info["class_counts"].values()) == len(y)


def test_warm_start_gives_same_model():
    X, y = toy_data(n=2000, seed=3)
    cold = OrbModel().fit(X, y)
    prev = OrbModel().fit(X[:1500], y[:1500])
    warm = OrbModel().fit(X, y, warm_start_from=prev)
    # ίδια κυρτή βελτιστοποίηση· διαφορές μόνο στο όριο της ανοχής σύγκλισης
    assert np.allclose(cold.predict_proba(X[:200]), warm.predict_proba(X[:200]), atol=2e-3)


# ---------------------------------------------------------------- 3. εξήγηση (συνεισφορές)

def test_contributions_explain_log_odds_tp_vs_sl():
    X, y = toy_data(n=1500, seed=5)
    m = OrbModel().fit(X, y)
    clf = m._est[-1]
    classes = list(clf.classes_)
    b = clf.intercept_[classes.index(LABEL_TP)] - clf.intercept_[classes.index(LABEL_SL)]
    for i in range(5):
        c = m.contributions(X[i])
        assert list(c) == FEATURE_COLUMNS
        p = m.predict_proba(X[i])[0]
        assert sum(c.values()) + b == pytest.approx(np.log(p[LABEL_TP] / p[LABEL_SL]), abs=1e-8)
    # το x0 κινεί το TP έναντι του SL (βλ. toy_data) ⇒ η μεγαλύτερη συνεισφορά
    big = X[np.argmax(np.abs(X[:, 0]))]
    c = m.contributions(big)
    assert max(c, key=lambda k: abs(c[k])) == FEATURE_COLUMNS[0]


def test_contributions_edge_cases():
    X, y = toy_data(classes=(0, 1))            # δύο καταστάσεις: δουλεύει
    m = OrbModel().fit(X, y)
    c = m.contributions(X[0])
    p = m.predict_proba(X[0])[0]
    clf = m._est[-1]
    assert sum(c.values()) + clf.intercept_[0] * (1 if list(clf.classes_) == [1, 0] else -1) == \
        pytest.approx(np.log(p[0] / p[1]), abs=1e-8)
    X, y = toy_data(classes=(0, 2))            # χωρίς SL ⇒ δεν ορίζεται
    assert OrbModel().fit(X, y).contributions(X[0]) == {}
    X, y = toy_data()
    assert OrbModel(kind="hgb").fit(X, y).contributions(X[0]) == {}
    m = OrbModel().fit(X, y)
    bad = X[0].copy()
    bad[1] = np.nan
    assert m.contributions(bad) == {}


# ---------------------------------------------------------------- 4. βρίσκει πραγματικό σήμα, όχι φανταστικό

@pytest.fixture(scope="module")
def planted_panel(synth_planted, small_universe, session_cfg):
    prices, index_df = synth_planted
    pn = features.build_panel(prices, index_df, small_universe, session_cfg)
    return pn, labels.triple_barrier(prices, pn, session_cfg)


def _auc(score, target):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(target, score)


def test_model_finds_planted_signal_but_not_noise(planted_panel, panel, labels_df, session_cfg):
    cfg = wf_cfg(session_cfg)
    pn, lab = planted_panel
    preds, _ = walk_forward(pn, lab, cfg)
    j = preds.join(lab["label"]).dropna(subset=["label"])
    auc_planted = _auc(j["p_tp"], j["label"] == LABEL_TP)
    preds0, _ = walk_forward(panel, labels_df, cfg)
    j0 = preds0.join(labels_df["label"]).dropna(subset=["label"])
    auc_noise = _auc(j0["p_tp"], j0["label"] == LABEL_TP)
    # planted_signal = 1: AUC ≈ 0,63 εκτός δείγματος· καθαρός θόρυβος: ≈ 0,5
    assert auc_planted > 0.58
    assert 0.44 < auc_noise < 0.56
