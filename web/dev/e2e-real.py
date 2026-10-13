"""Έλεγχος από άκρη σε άκρη: το πραγματικό UI απέναντι στον ΠΡΑΓΜΑΤΙΚΟ διακομιστή (headless Chromium).

Σε αντίθεση με το ui-check.py (mock διακομιστής), εδώ τρέχει όλη η εφαρμογή:
`python -m athex_orb demo` σε προσωρινό φάκελο δεδομένων, `python -m athex_orb serve`, και ο browser.

Ελέγχει:
* όλες τις σελίδες σε 1440×900 και 390×844, σκούρο & φωτεινό θέμα: χωρίς σφάλματα κονσόλας,
  χωρίς αποτυχημένα αιτήματα, χωρίς οριζόντια κύλιση·
* αναζήτηση «οπαπ» + Enter, αστέρι/ξε‑αστέρι με επιμονή μετά από ανανέωση, προσθήκη μετοχής χρήστη
  (INLOT → συνθετικά δεδομένα στην επίδειξη) και αφαίρεσή της·
* ρυθμίσεις: αποθήκευση, γρήγορος επανυπολογισμός σημάτων, ένδειξη «απαιτείται νέα ενημέρωση»·
* το κουμπί «Ενημέρωση»: πρόοδος μέχρι το τέλος. Στην επίδειξη, χωρίς διαδίκτυο, η λήψη από το Yahoo
  αποτυγχάνει ⇒ αναμένεται ελληνικό μήνυμα σφάλματος και τα δεδομένα επίδειξης μένουν ως έχουν·
* δεύτερη εγκατάσταση με πηγή «csv» (data/csv): από την κάρτα έναρξης «Λήψη πραγματικών δεδομένων»
  και μετά ξανά από το κουμπί «Ενημέρωση» — η εργασία ολοκληρώνεται κανονικά.

Αποθηκεύει τα στιγμιότυπα του docs/screenshots/ (overview-desktop.png κ.λπ.).

Χρήση:
    python web/dev/e2e-real.py              # όλα, σε προσωρινούς φακέλους (δεν αγγίζει το data/)
    python web/dev/e2e-real.py --no-shots   # χωρίς στιγμιότυπα
"""
from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHOTS = ROOT / "docs" / "screenshots"
ROUTES = [
    ("overview", "#/"),
    ("stock", "#/stock/PPC.AT"),
    ("signals", "#/signals"),
    ("backtest", "#/backtest"),
    ("journal", "#/journal"),
    ("settings", "#/settings"),
]
VIEWPORTS = [("desktop", 1440, 900), ("mobile", 390, 844)]
IGNORED = ("fonts.googleapis.com", "fonts.gstatic.com")   # προαιρετικές γραμματοσειρές (χωρίς διαδίκτυο αποτυγχάνουν)
FAVS = ["OPAP.AT", "PPC.AT", "ETE.AT", "MTLN.AT"]

failures: list[str] = []


def check(cond: bool, msg: str) -> bool:
    print(("  ✓ " if cond else "  ✕ ") + msg, flush=True)
    if not cond:
        failures.append(msg)
    return cond


# ---------------------------------------------------------------- διακομιστές

def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def cli(*args: str, log=None) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "athex_orb", *args], cwd=ROOT, check=True,
                          stdout=log or subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def start_server(data_dir: Path, port: int, config: Path | None = None, log: Path | None = None) -> subprocess.Popen:
    args = [sys.executable, "-m", "athex_orb", "--data-dir", str(data_dir)]
    if config is not None:
        args += ["--config", str(config)]
    out = open(log, "w") if log else subprocess.DEVNULL
    proc = subprocess.Popen([*args, "serve", "--port", str(port)], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
    for _ in range(150):
        if port_open(port):
            return proc
        time.sleep(0.1)
    proc.terminate()
    raise RuntimeError(f"ο διακομιστής στη θύρα {port} δεν ξεκίνησε")


def api(base: str, path: str, method: str = "GET", body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, method=method, data=data,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


# ---------------------------------------------------------------- browser

class Watch:
    """Σφάλματα κονσόλας/σελίδας και αποτυχημένα αιτήματα μιας σελίδας."""

    def __init__(self, page, allow_status=()):
        self.errors: list[str] = []
        self.aborted = 0
        self.allow_status = set(allow_status)
        page.on("console", self._console)
        page.on("pageerror", lambda e: self.errors.append(f"pageerror: {e}"))
        page.on("requestfailed", self._failed)
        page.on("response", self._response)

    def _console(self, m):
        if m.type == "error" and not any(x in m.text + (m.location or {}).get("url", "") for x in IGNORED):
            self.errors.append(f"console: {m.text}")

    def _failed(self, r):
        if any(x in r.url for x in IGNORED):
            return
        if "ERR_ABORTED" in (r.failure or ""):      # ακυρώσεις κατά την αλλαγή σελίδας (AbortController)
            self.aborted += 1
            return
        self.errors.append(f"requestfailed: {r.url} {r.failure}")

    def _response(self, r):
        if r.status >= 400 and r.status not in self.allow_status and not any(x in r.url for x in IGNORED):
            self.errors.append(f"HTTP {r.status}: {r.url}")


def new_page(browser, w, h, theme="dark", allow_status=()):
    ctx = browser.new_context(viewport={"width": w, "height": h}, color_scheme=theme, device_scale_factor=1,
                              has_touch=w < 760)
    ctx.add_init_script(f"try{{localStorage.setItem('orb.theme', JSON.stringify('{theme}'))}}catch(e){{}}")
    page = ctx.new_page()
    return ctx, page, Watch(page, allow_status)


def wait_ready(page, timeout=20000):
    page.wait_for_function(
        """() => { const v = document.querySelector('#view .view');
                 return v && v.children.length && !v.querySelector('[aria-busy="true"]')
                        && !document.querySelector('.sb-list .sk-row'); }""", timeout=timeout)
    page.wait_for_timeout(400)


def no_overflow(page):
    sw = page.evaluate("[document.documentElement.scrollWidth, window.innerWidth]")
    return sw[0] <= sw[1], sw


def shot(page, name: str, enabled: bool, full: bool = True) -> None:
    if not enabled:
        return
    path = SHOTS / f"{name}.png"
    page.screenshot(path=str(path), full_page=full)
    try:  # μικρότερα αρχεία: παλέτα 256 χρωμάτων (octree χωρίς dithering: κρατά σωστά τα χρώματα σήμανσης)
        from PIL import Image
        im = Image.open(path).convert("RGB")
        im.quantize(colors=256, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE).save(path, optimize=True)
    except Exception:
        pass
    print(f"  · στιγμιότυπο {path.relative_to(ROOT)}", flush=True)


def watch_job_progress(page) -> None:
    """Καταγράφει τα ποσοστά που δείχνει το κουμπί εργασιών (για να φανεί ότι η πρόοδος κινείται)."""
    page.evaluate("""() => { window.__prog = []; const b = document.getElementById('job-btn');
        new MutationObserver(() => { const p = b.querySelector('.job-pct');
          if (p) { const v = parseInt(p.textContent, 10); if (!Number.isNaN(v) && window.__prog[window.__prog.length - 1] !== v) window.__prog.push(v); } })
          .observe(b, { childList: true, subtree: true, characterData: true }); }""")


def wait_job_end(page, timeout_s: float) -> None:
    page.wait_for_function("document.getElementById('job-btn').classList.contains('is-running')", timeout=8000)
    page.wait_for_function("!document.getElementById('job-btn').classList.contains('is-running')",
                           timeout=int(timeout_s * 1000))


# ---------------------------------------------------------------- σενάρια

def run_pages(browser, base: str, shots: bool) -> None:
    for vp, w, h in VIEWPORTS:
        for theme in ("dark", "light"):
            print(f"\n[σελίδες · {vp} {w}×{h} · {theme}]", flush=True)
            ctx, page, watch = new_page(browser, w, h, theme)
            for name, route in ROUTES:
                page.goto(f"{base}/{route}")
                wait_ready(page)
                ok, sw = no_overflow(page)
                check(ok, f"{name}: χωρίς οριζόντια κύλιση {sw}")
                if name == "stock":
                    check(page.locator(".orb-card .orb").count() == 1 and page.locator(".agent").count() == 6,
                          "stock: Σφαίρα + έξι πράκτορες")
                    check(page.locator('[data-chart="price"] svg').count() == 1, "stock: γράφημα τιμής")
                if name == "backtest":
                    check(page.locator(".verdict .v-text").inner_text().strip() != "", "backtest: κείμενο ετυμηγορίας")
                if theme == "dark" and vp == "desktop" and name in ("overview", "stock", "backtest", "signals"):
                    shot(page, f"{name}-desktop", shots)
                if theme == "light" and vp == "desktop" and name == "overview":
                    shot(page, "overview-light", shots)
                if theme == "dark" and vp == "mobile" and name == "stock":
                    shot(page, "stock-mobile", shots)
            check(not watch.errors, f"χωρίς σφάλματα κονσόλας/αιτημάτων ({len(watch.errors)}) {watch.errors[:3]}")
            ctx.close()


def run_interactions(browser, base: str) -> None:
    print("\n[αναζήτηση, αγαπημένα, προσθήκη μετοχής · desktop]", flush=True)
    for t in api(base, "/api/favorites")["tickers"]:
        api(base, f"/api/favorites/{t}", "DELETE")
    ctx, page, watch = new_page(browser, 1440, 900, "dark")
    page.goto(f"{base}/#/")
    wait_ready(page)
    page.keyboard.press("/")
    check(page.evaluate("document.activeElement.id") == "sb-search", "το «/» εστιάζει την αναζήτηση")
    page.keyboard.type("οπαπ")
    page.wait_for_timeout(450)
    first = page.locator("#sb-list .stock-row").first.get_attribute("data-ticker")
    check(first == "OPAP.AT", f"«οπαπ» → OPAP.AT (πρώτο: {first})")
    page.keyboard.press("Enter")
    page.wait_for_function("location.hash === '#/stock/OPAP.AT'", timeout=5000)
    wait_ready(page)
    check(page.locator(".stock-head h1").inner_text().startswith("ΟΠΑΠ"), "Enter ανοίγει τη σελίδα του ΟΠΑΠ")
    check(page.locator(".decision").count() == 1, "λωρίδα απόφασης στη σελίδα μετοχής")

    # αστέρι από τη σελίδα μετοχής
    page.click("[data-stock-fav]")
    page.wait_for_timeout(500)
    check(page.locator("[data-stock-fav]").get_attribute("aria-pressed") == "true", "★ στη σελίδα μετοχής (αισιόδοξα)")
    check("OPAP.AT" in api(base, "/api/favorites")["tickers"], "ο διακομιστής αποθήκευσε το αγαπημένο")
    page.reload()
    wait_ready(page)
    check(page.locator("[data-stock-fav]").get_attribute("aria-pressed") == "true", "το ★ μένει μετά από ανανέωση")
    page.click('.sb-tabs [data-tab="fav"]')
    page.wait_for_timeout(250)
    check(page.locator('#sb-list .stock-row[data-ticker="OPAP.AT"]').count() == 1, "ο ΟΠΑΠ εμφανίζεται στα «★ Αγαπημένα»")
    # ξε‑αστέρι από τη λίστα και ανανέωση
    page.click('#sb-list .stock-row[data-ticker="OPAP.AT"] .star')
    page.wait_for_timeout(600)
    check("OPAP.AT" not in api(base, "/api/favorites")["tickers"], "κλικ στο ★ της λίστας αφαιρεί το αγαπημένο")
    page.reload()
    wait_ready(page)
    check(page.locator("[data-stock-fav]").get_attribute("aria-pressed") == "false", "η αφαίρεση μένει μετά από ανανέωση")
    # Shift+Enter από την αναζήτηση
    page.fill("#sb-search", "δεη")
    page.wait_for_timeout(400)
    page.keyboard.press("Shift+Enter")
    page.wait_for_timeout(600)
    check("PPC.AT" in api(base, "/api/favorites")["tickers"], "«δεη» + Shift+Enter → ★ στη ΔΕΗ")
    page.keyboard.press("Escape")

    # προσθήκη μετοχής χρήστη
    page.click("#sb-search")
    page.fill("#sb-search", "INLOT")
    page.wait_for_timeout(500)
    add = page.locator("#sb-list .add-row")
    if check(add.count() == 1, "για άγνωστο σύμβολο εμφανίζεται «Προσθήκη INLOT.AT»"):
        t0 = time.perf_counter()
        add.click()
        page.wait_for_function("location.hash === '#/stock/INLOT.AT'", timeout=30000)
        wait_ready(page)
        dt = time.perf_counter() - t0
        check(page.locator('[data-chart="price"] svg').count() == 1, f"η INLOT.AT έχει δεδομένα και γράφημα ({dt:.1f}s)")
        check("INLOT.AT" in api(base, "/api/favorites")["tickers"], "η νέα μετοχή μπήκε στα αγαπημένα")
        item = next((i for i in api(base, "/api/universe")["items"] if i["ticker"] == "INLOT.AT"), {})
        check(item.get("custom") is True and item.get("has_data") is True and item.get("orb") is not None,
              "η INLOT.AT έχει σφαίρα με το ήδη εκπαιδευμένο μοντέλο")
        page.once("dialog", lambda d: d.accept())
        page.click("[data-remove]")
        page.wait_for_function("location.hash === '#/'", timeout=10000)
        wait_ready(page)
        check(all(i["ticker"] != "INLOT.AT" for i in api(base, "/api/universe")["items"]), "αφαίρεση μετοχής χρήστη")
    check(not watch.errors, f"χωρίς σφάλματα κονσόλας/αιτημάτων ({len(watch.errors)}) {watch.errors[:3]}")
    ctx.close()


def run_mobile_search(browser, base: str, shots: bool) -> None:
    print("\n[αναζήτηση · κινητό]", flush=True)
    ctx, page, watch = new_page(browser, 390, 844, "dark")
    page.goto(f"{base}/#/")
    wait_ready(page)
    page.click("#open-sidebar")
    page.wait_for_timeout(400)
    check(page.evaluate("document.activeElement.id") == "sb-search", "το συρτάρι ανοίγει με εστίαση στην αναζήτηση")
    page.keyboard.type("οπαπ")
    page.wait_for_timeout(500)
    ok, sw = no_overflow(page)
    check(ok, f"συρτάρι: χωρίς οριζόντια κύλιση {sw}")
    shot(page, "search-mobile", shots, full=False)
    page.keyboard.press("Enter")
    page.wait_for_function("location.hash === '#/stock/OPAP.AT'", timeout=5000)
    check(not page.evaluate("document.getElementById('sidebar').classList.contains('is-open')"), "το συρτάρι κλείνει μετά την επιλογή")
    wait_ready(page)
    check(not watch.errors, f"χωρίς σφάλματα κονσόλας/αιτημάτων ({len(watch.errors)}) {watch.errors[:3]}")
    ctx.close()


def run_settings(browser, base: str) -> None:
    print("\n[ρυθμίσεις: αποθήκευση, γρήγορος επανυπολογισμός σημάτων, ένδειξη επανεκπαίδευσης]", flush=True)
    ctx, page, watch = new_page(browser, 1440, 900, "dark")
    page.goto(f"{base}/#/settings")
    wait_ready(page)
    takes0 = api(base, "/api/signals")["counts"]["take"]

    def save():
        page.click("[data-save]")
        page.wait_for_selector(".toast.success", timeout=20000)
        wait_ready(page)
        page.evaluate("document.querySelectorAll('.toast').forEach((t) => t.remove())")

    page.fill("#f-guard-min_ev_net", "-1,5")
    page.wait_for_timeout(150)
    check(page.locator("[data-save]").is_enabled(), "η αλλαγή ενεργοποιεί την «Αποθήκευση»")
    t0 = time.perf_counter()
    save()
    st, sig = api(base, "/api/status"), api(base, "/api/signals")
    check(abs(api(base, "/api/settings")["guard"]["min_ev_net"] + 0.015) < 1e-12 and not st["retrain_needed"],
          f"ελάχιστο EV −1,5% αποθηκεύτηκε χωρίς ανάγκη επανεκπαίδευσης ({time.perf_counter() - t0:.1f}s)")
    check(sig["counts"]["take"] > takes0, f"τα σήματα ξαναϋπολογίστηκαν: υποψήφιες αγορές {takes0} → {sig['counts']['take']}")
    page.goto(f"{base}/#/")
    wait_ready(page)
    kpi = page.locator(".kpi").nth(1).locator(".kpi-value").inner_text().strip()
    check(kpi == str(sig["counts"]["take"]), f"η «Αγορά» δείχνει αμέσως τις νέες υποψήφιες ({kpi})")
    page.goto(f"{base}/#/settings")
    wait_ready(page)
    page.fill("#f-guard-min_ev_net", "0,30")
    save()
    check(api(base, "/api/signals")["counts"]["take"] == takes0, "επαναφορά ελάχιστου EV → ίδια σήματα με πριν")
    # όρια εξόδου: χρειάζεται νέα ενημέρωση· η επαναφορά τη «σβήνει»
    page.fill("#f-barriers-k_tp", "2,5")
    save()
    check(api(base, "/api/status")["retrain_needed"] and page.locator(".save-bar .chip-warn").count() == 1,
          "αλλαγή στόχου (k_tp 2,5) → «Απαιτείται νέα ενημέρωση»")
    page.fill("#f-barriers-k_tp", "2")
    save()
    check(not api(base, "/api/status")["retrain_needed"] and page.locator(".save-bar .chip-warn").count() == 0,
          "επαναφορά k_tp → δεν χρειάζεται πια ενημέρωση")
    page.fill("#f-barriers-k_tp", "abc")
    page.wait_for_timeout(150)
    check(not page.locator("[data-save]").is_enabled() and page.locator(".field.is-invalid").count() == 1,
          "μη έγκυρη τιμή → μήνυμα στο πεδίο, χωρίς αποθήκευση")
    check(not watch.errors, f"χωρίς σφάλματα κονσόλας/αιτημάτων ({len(watch.errors)}) {watch.errors[:3]}")
    ctx.close()


def run_demo_refresh(browser, base: str) -> None:
    print("\n[κουμπί «Ενημέρωση» σε δεδομένα επίδειξης]", flush=True)
    before = api(base, "/api/status")
    ctx, page, watch = new_page(browser, 1440, 900, "dark")
    page.goto(f"{base}/#/")
    wait_ready(page)
    watch_job_progress(page)
    t0 = time.perf_counter()
    page.click("#job-btn")
    wait_job_end(page, 240)
    dt = time.perf_counter() - t0
    prog = page.evaluate("window.__prog")
    check(len(prog) >= 2, f"η πρόοδος εμφανίζεται στο κουμπί ({prog[:8]}…)")
    after = api(base, "/api/status")
    job = api(base, "/api/jobs")["items"][0]
    if job["status"] == "error":
        # χωρίς διαδίκτυο: σωστό σφάλμα, τίποτα δεν χάθηκε
        check(page.locator(".toast.error").count() >= 1, f"ειδοποίηση σφάλματος ({dt:.0f}s): {job['message_el'][:90]}")
        check("Yahoo" in job["message_el"] and after["mode"] == before["mode"] == "demo",
              "η αποτυχία λήψης δεν αλλάζει τα δεδομένα επίδειξης")
    else:
        check(job["status"] == "done" and page.locator(".toast.success").count() >= 1,
              f"η ενημέρωση ολοκληρώθηκε ({dt:.0f}s) · mode={after['mode']}")
    wait_ready(page)
    check(not watch.errors, f"χωρίς σφάλματα κονσόλας/αιτημάτων ({len(watch.errors)}) {watch.errors[:3]}")
    ctx.close()


def run_csv_refresh(browser, tmp: Path, demo_data: Path) -> None:
    """Πηγή «csv»: τα αρχεία του χρήστη στο data/csv. Η ενημέρωση από το UI πρέπει να ολοκληρώνεται."""
    print("\n[κουμπί «Ενημέρωση» με πηγή csv (πραγματικά δεδομένα από αρχεία)]", flush=True)
    data = tmp / "csv-data"
    (data / "csv").mkdir(parents=True)
    for f in (demo_data / "prices").glob("*.csv"):
        shutil.copy(f, data / "csv" / f.name)
    cfgp = tmp / "csv-config.toml"
    cfgp.write_text((ROOT / "config.toml").read_text(encoding="utf-8").replace('source = "yahoo"', 'source = "csv"'),
                    encoding="utf-8")
    port = free_port()
    proc = start_server(data, port, cfgp, log=tmp / "csv-server.log")
    base = f"http://127.0.0.1:{port}"
    try:
        ctx, page, watch = new_page(browser, 1440, 900, "light")
        page.goto(f"{base}/#/")
        wait_ready(page)
        check(page.locator(".onboard [data-job='refresh']").count() == 1, "χωρίς δεδομένα → κάρτα έναρξης")
        watch_job_progress(page)
        t0 = time.perf_counter()
        page.click(".onboard [data-job='refresh']")
        wait_job_end(page, 240)
        page.wait_for_selector(".kpis", timeout=20000)
        wait_ready(page)
        st = api(base, "/api/status")
        check(st["mode"] == "live" and st["n_with_data"] > 0,
              f"«Λήψη πραγματικών δεδομένων» ολοκληρώθηκε ({time.perf_counter() - t0:.0f}s, "
              f"{st['n_with_data']}/{st['n_tickers']} μετοχές, πρόοδος {page.evaluate('window.__prog')[:6]}…)")
        check(page.locator("#demo-badge").is_hidden(), "χωρίς ένδειξη ΕΠΙΔΕΙΞΗ σε πραγματικά δεδομένα")
        watch_job_progress(page)
        t0 = time.perf_counter()
        page.click("#job-btn")
        wait_job_end(page, 240)
        prog = page.evaluate("window.__prog")
        job = api(base, "/api/jobs")["items"][0]
        check(job["type"] == "refresh" and job["status"] == "done" and page.locator(".toast.success").count() >= 1,
              f"«Ενημέρωση» από το κουμπί: ολοκληρώθηκε σε {time.perf_counter() - t0:.0f}s (πρόοδος {prog[:5]}…{prog[-2:]})")
        wait_ready(page)
        check(not watch.errors, f"χωρίς σφάλματα κονσόλας/αιτημάτων ({len(watch.errors)}) {watch.errors[:3]}")
        ctx.close()
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-shots", action="store_true", help="χωρίς στιγμιότυπα")
    ap.add_argument("--skip-refresh", action="store_true", help="χωρίς τους ελέγχους εργασίας «Ενημέρωση»")
    ap.add_argument("--keep", action="store_true", help="κράτα τους προσωρινούς φακέλους")
    args = ap.parse_args()
    shots = not args.no_shots

    from playwright.sync_api import sync_playwright

    tmp = Path(tempfile.mkdtemp(prefix="athex-orb-e2e-"))
    demo_data = tmp / "demo-data"
    print(f"Προσωρινός φάκελος: {tmp}", flush=True)
    t0 = time.perf_counter()
    cli("--data-dir", str(demo_data), "demo")
    print(f"python -m athex_orb demo: {time.perf_counter() - t0:.1f}s", flush=True)
    port = free_port()
    proc = start_server(demo_data, port, log=tmp / "demo-server.log")
    base = f"http://127.0.0.1:{port}"
    if shots:
        SHOTS.mkdir(parents=True, exist_ok=True)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            run_interactions(browser, base)
            run_settings(browser, base)
            for t in api(base, "/api/favorites")["tickers"]:
                api(base, f"/api/favorites/{t}", "DELETE")
            for t in FAVS:                                  # αγαπημένα για τα στιγμιότυπα
                api(base, f"/api/favorites/{t}", "PUT")
            run_mobile_search(browser, base, shots)
            run_pages(browser, base, shots)
            if not args.skip_refresh:
                run_demo_refresh(browser, base)
                run_csv_refresh(browser, tmp, demo_data)
            browser.close()
    finally:
        proc.terminate()
        proc.wait(timeout=10)
        if not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{'ΟΛΑ ΚΑΛΑ' if not failures else f'{len(failures)} ΑΠΟΤΥΧΙΕΣ'} · {time.perf_counter() - t0:.0f}s")
    for f in failures:
        print("  ✕", f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
