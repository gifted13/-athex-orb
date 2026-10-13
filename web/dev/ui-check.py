"""Έλεγχος του UI σε headless Chromium (playwright) απέναντι στον mock διακομιστή.

Ελέγχει: σφάλματα κονσόλας, οριζόντια υπερχείλιση, όλες τις σελίδες σε 1440×900 και 390×844
(σκούρο & φωτεινό θέμα), αναζήτηση με πληκτρολόγιο (χωρίς τόνους), αγαπημένα (και ότι μένουν
μετά από ανανέωση), προσθήκη μετοχής, tooltip γραφήματος, συρτάρι στο κινητό, κατάσταση «χωρίς
δεδομένα» και «διακομιστής εκτός». Αποθηκεύει στιγμιότυπα στο docs/screenshots/dev-*.png.

Χρήση:
    python web/dev/ui-check.py            # ξεκινά μόνος του τους mock servers (8765/8766/8767)
    python web/dev/ui-check.py --no-shots # μόνο έλεγχοι
"""
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MOCK = ROOT / "web" / "dev" / "mock-server.py"
# στιγμιότυπα του mock σε προσωρινό φάκελο· τα στιγμιότυπα του docs/screenshots/ βγαίνουν από τον
# πραγματικό διακομιστή με το web/dev/e2e-real.py
SHOTS = Path(tempfile.gettempdir()) / "athex-orb-ui-check"

ROUTES = [
    ("overview", "#/"),
    ("stock", "#/stock/OPAP.AT"),
    ("signals", "#/signals"),
    ("backtest", "#/backtest"),
    ("journal", "#/journal"),
    ("settings", "#/settings"),
]
VIEWPORTS = [("desktop", 1440, 900), ("mobile", 390, 844)]
IGNORED_CONSOLE = ("fonts.googleapis.com", "fonts.gstatic.com")

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  ✓ " if cond else "  ✕ ") + msg)
    if not cond:
        failures.append(msg)


def port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_mock(port: int, *extra: str, reuse: bool = True) -> subprocess.Popen | None:
    if reuse and port_open(port):
        return None
    proc = subprocess.Popen([sys.executable, str(MOCK), "--port", str(port), *extra],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
        if port_open(port):
            return proc
        time.sleep(0.1)
    raise RuntimeError(f"ο mock server στη θύρα {port} δεν ξεκίνησε")


def api(base: str, path: str, method: str = "GET"):
    req = urllib.request.Request(base + path, method=method)
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read().decode())


def new_page(browser, w, h, theme, errors):
    ctx = browser.new_context(viewport={"width": w, "height": h}, color_scheme=theme, device_scale_factor=1,
                              has_touch=w < 760, is_mobile=False)
    ctx.add_init_script(f"try{{localStorage.setItem('orb.theme', JSON.stringify('{theme}'))}}catch(e){{}}")
    page = ctx.new_page()

    def on_console(m):
        if m.type == "error" and not any(x in (m.location or {}).get("url", "") + m.text for x in IGNORED_CONSOLE):
            errors.append(f"console: {m.text}")
    page.on("console", on_console)
    page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
    return ctx, page


def wait_ready(page, timeout=8000):
    page.wait_for_function(
        """() => { const v = document.querySelector('#view .view');
                 return v && v.children.length && !v.querySelector('[aria-busy="true"]') && !document.querySelector('.sb-list .sk-row'); }""",
        timeout=timeout)
    page.wait_for_timeout(450)


def no_overflow(page) -> tuple[bool, list[int]]:
    sw = page.evaluate("[document.documentElement.scrollWidth, window.innerWidth]")
    return sw[0] <= sw[1], sw


def shot(page, name, full=True, shots=True):
    if not shots:
        return
    path = SHOTS / f"dev-{name}.png"
    page.screenshot(path=str(path), full_page=full)
    try:  # μικρότερα αρχεία: παλέτα 256 χρωμάτων με dithering
        from PIL import Image
        im = Image.open(path).convert("RGB")
        im.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG).save(path, optimize=True)
    except Exception:
        pass


def run(base: str, base_empty: str, base_down: str, shots: bool) -> None:
    from playwright.sync_api import sync_playwright

    SHOTS.mkdir(parents=True, exist_ok=True)
    # καθαρή αρχική κατάσταση (αγαπημένα, μετοχές χρήστη)
    try:
        api(base, "/api/universe/INTEK.AT", "DELETE")
    except Exception:
        pass
    for t in api(base, "/api/favorites")["tickers"]:
        api(base, f"/api/favorites/{t}", "DELETE")
    for t in ["OPAP.AT", "PPC.AT", "ETE.AT", "MTLN.AT"]:
        api(base, f"/api/favorites/{t}", "PUT")

    with sync_playwright() as p:
        browser = p.chromium.launch()

        # 1) Όλες οι σελίδες, σε δύο μεγέθη και δύο θέματα
        for vp, w, h in VIEWPORTS:
            for theme in ("dark", "light"):
                print(f"\n[{vp} {w}×{h} · {theme}]")
                errors: list[str] = []
                ctx, page = new_page(browser, w, h, theme, errors)
                for name, route in ROUTES:
                    page.goto(base + "/" + route)
                    wait_ready(page)
                    ok, sw = no_overflow(page)
                    check(ok, f"{name}: χωρίς οριζόντια κύλιση {sw}")
                    check(page.evaluate("document.documentElement.dataset.theme") == theme, f"{name}: θέμα {theme}")
                    shot(page, f"{name}-{vp}-{theme}", full=(theme == "dark"), shots=shots)
                check(not errors, f"χωρίς σφάλματα κονσόλας ({len(errors)}) {errors[:3]}")
                ctx.close()

        # 2) Αναζήτηση, πληκτρολόγιο, αγαπημένα (desktop)
        print("\n[αλληλεπίδραση · desktop]")
        errors = []
        ctx, page = new_page(browser, 1440, 900, "dark", errors)
        page.goto(base + "/#/")
        wait_ready(page)
        page.keyboard.press("/")
        check(page.evaluate("document.activeElement.id") == "sb-search", "το «/» εστιάζει την αναζήτηση")
        page.keyboard.type("οπαπ")
        page.wait_for_timeout(250)
        first = page.locator("#sb-list .stock-row").first.get_attribute("data-ticker")
        check(first == "OPAP.AT", f"«οπαπ» → OPAP.AT (βρέθηκε {first})")
        page.fill("#sb-search", "πειραιως")
        page.wait_for_timeout(250)
        check(page.locator('#sb-list .stock-row[data-ticker="TPEIR.AT"]').count() == 1, "«πειραιως» (χωρίς τόνους) → Τράπεζα Πειραιώς")
        page.fill("#sb-search", "τραπεζες")
        page.wait_for_timeout(250)
        n_banks = page.locator("#sb-list .stock-row[data-ticker]").count()
        check(n_banks >= 6, f"«τραπεζες» → κλάδος Τράπεζες ({n_banks} αποτελέσματα)")
        page.keyboard.press("ArrowDown")
        page.keyboard.press("ArrowDown")
        act = page.evaluate("document.getElementById('sb-search').getAttribute('aria-activedescendant')")
        check(act == "sbo-2", f"↓↓ μετακινεί την επιλογή (aria-activedescendant={act})")
        page.keyboard.press("ArrowUp")
        target = page.locator("#sb-list .stock-row.is-active").get_attribute("data-ticker")
        page.keyboard.press("Enter")
        page.wait_for_timeout(300)
        check(page.evaluate("location.hash") == f"#/stock/{target}", f"Enter ανοίγει τη μετοχή {target}")
        wait_ready(page)
        # αστέρι με Shift+Enter (Alpha Bank δεν είναι στα αγαπημένα)
        page.fill("#sb-search", "alpha")
        page.wait_for_timeout(300)
        row = page.locator('#sb-list .stock-row[data-ticker="ALPHA.AT"]')
        check(row.count() == 1, "«alpha» → ALPHA.AT")
        idx = int(row.get_attribute("data-i"))
        for _ in range(idx):
            page.keyboard.press("ArrowDown")
        page.keyboard.press("Shift+Enter")
        page.wait_for_timeout(400)
        pressed = page.locator('#sb-list .stock-row[data-ticker="ALPHA.AT"] .star').get_attribute("aria-pressed")
        check(pressed == "true", "Shift+Enter → αστέρι ενεργό (αισιόδοξη ενημέρωση)")
        check(page.locator(".toast").count() >= 1, "εμφανίζεται ειδοποίηση «Προστέθηκε στα αγαπημένα»")
        check("ALPHA.AT" in api(base, "/api/favorites")["tickers"], "ο διακομιστής αποθήκευσε το αγαπημένο")
        page.keyboard.press("Escape")
        check(page.input_value("#sb-search") == "", "Esc καθαρίζει την αναζήτηση")
        # επιμονή μετά από ανανέωση
        page.reload()
        wait_ready(page)
        page.click('.sb-tabs [data-tab="fav"]')
        page.wait_for_timeout(200)
        check(page.locator('#sb-list .stock-row[data-ticker="ALPHA.AT"]').count() == 1, "το αγαπημένο παραμένει μετά από ανανέωση")
        stored = page.evaluate("JSON.parse(localStorage.getItem('orb.favorites')||'[]')")
        check("ALPHA.AT" in stored, "τα αγαπημένα αντιγράφονται και στο localStorage")
        # αφαίρεση με κλικ στο αστέρι
        page.click('#sb-list .stock-row[data-ticker="ALPHA.AT"] .star')
        page.wait_for_timeout(400)
        check("ALPHA.AT" not in api(base, "/api/favorites")["tickers"], "κλικ στο ★ αφαιρεί από τα αγαπημένα")
        # αποτυχία διακομιστή → επαναφορά (rollback) + ειδοποίηση σφάλματος
        page.route("**/api/favorites/KRI.AT", lambda route: route.fulfill(status=500, body='{"detail":"Δοκιμαστικό σφάλμα"}', content_type="application/json"))
        page.click('.sb-tabs [data-tab="all"]')
        page.wait_for_timeout(150)
        page.evaluate("""() => { window.__kri = []; const list = document.getElementById('sb-list');
            new MutationObserver(() => { const b = list.querySelector('[data-ticker="KRI.AT"] .star');
              if (b) window.__kri.push(b.getAttribute('aria-pressed')); }).observe(list, { childList: true, subtree: true, attributes: true }); }""")
        page.click('#sb-list .stock-row[data-ticker="KRI.AT"] .star')
        page.wait_for_timeout(700)
        seq = page.evaluate("window.__kri")
        rolled = page.locator('#sb-list .stock-row[data-ticker="KRI.AT"] .star').get_attribute("aria-pressed")
        check("true" in seq and rolled == "false", f"σφάλμα αποθήκευσης → αισιόδοξο ★ και επαναφορά ({seq[:4]} → {rolled})")
        check(page.locator(".toast.error").count() >= 1, "σφάλμα αποθήκευσης → ειδοποίηση σφάλματος")
        page.unroute("**/api/favorites/KRI.AT")
        errors[:] = [e for e in errors if "500" not in e]
        # προσθήκη νέας μετοχής
        page.click("#sb-search")
        page.fill("#sb-search", "INTEK")
        page.wait_for_timeout(450)
        add = page.locator("#sb-list .add-row")
        check(add.count() == 1, "για άγνωστο σύμβολο εμφανίζεται «Προσθήκη INTEK.AT»")
        if add.count():
            add.click()
            page.wait_for_function("location.hash === '#/stock/INTEK.AT'", timeout=5000)
            wait_ready(page)
            check("INTEK.AT" in api(base, "/api/favorites")["tickers"], "η νέα μετοχή μπήκε στα αγαπημένα")
            shot(page, "stock-custom-desktop-dark", full=False, shots=shots)
        # tooltip γραφήματος
        page.goto(base + "/#/stock/OPAP.AT")
        wait_ready(page)
        chart = page.locator('[data-chart="price"] svg')
        chart.scroll_into_view_if_needed()
        box = chart.bounding_box()
        page.mouse.move(box["x"] + box["width"] * 0.6, box["y"] + box["height"] * 0.5)
        page.wait_for_timeout(200)
        check(page.locator('[data-chart="price"] .ch-tip:not([hidden])').count() == 1, "tooltip στο γράφημα τιμής")
        shot(page, "chart-tooltip-desktop-dark", full=False, shots=shots)
        page.click('[data-mode="line"]')
        page.wait_for_timeout(200)
        check(page.locator('[data-mode="line"]').get_attribute("aria-pressed") == "true", "εναλλαγή σε γράφημα γραμμής")
        page.click('[data-mode="candles"]')
        # αστέρι στη σελίδα μετοχής
        before = page.locator("[data-stock-fav]").get_attribute("aria-pressed")
        page.click("[data-stock-fav]")
        page.wait_for_timeout(300)
        after = page.locator("[data-stock-fav]").get_attribute("aria-pressed")
        check(before != after, "κουμπί ★ στη σελίδα μετοχής")
        page.click("[data-stock-fav]")
        page.wait_for_timeout(300)
        # εναλλαγή θέματος και επιμονή
        page.click("#theme-btn")
        check(page.evaluate("document.documentElement.dataset.theme") == "light", "εναλλαγή θέματος")
        ctx2_theme = page.evaluate("localStorage.getItem('orb.theme')")
        check(ctx2_theme == '"light"', "το θέμα αποθηκεύεται")
        page.click("#theme-btn")
        # εργασία ενημέρωσης
        page.click("#job-btn")
        page.wait_for_timeout(1500)
        check("is-running" in (page.locator("#job-btn").get_attribute("class") or ""), "το «Ενημέρωση» δείχνει πρόοδο")
        shot(page, "job-progress-desktop-dark", full=False, shots=shots)
        page.wait_for_function("!document.querySelector('#job-btn').classList.contains('is-running')", timeout=15000)
        check(True, "η εργασία ολοκληρώθηκε και το κουμπί επανήλθε")
        # ρυθμίσεις: αλλαγή & αποθήκευση
        page.goto(base + "/#/settings")
        wait_ready(page)
        page.fill("#f-barriers-k_tp", "2,5")
        page.wait_for_timeout(150)
        check(page.locator("[data-save]").is_enabled(), "η αλλαγή ενεργοποιεί την αποθήκευση")
        page.fill("#f-barriers-k_tp", "abc")
        page.wait_for_timeout(150)
        check(not page.locator("[data-save]").is_enabled(), "μη έγκυρη τιμή → απενεργοποιημένη αποθήκευση")
        page.fill("#f-barriers-k_tp", "2,5")
        page.click("[data-save]")
        page.wait_for_timeout(500)
        check(abs(api(base, "/api/settings")["barriers"]["k_tp"] - 2.5) < 1e-9, "οι ρυθμίσεις αποθηκεύτηκαν (k_tp = 2,5)")
        shot(page, "settings-saved-desktop-dark", full=False, shots=shots)
        page.fill("#f-barriers-k_tp", "2")
        page.click("[data-save]")
        page.wait_for_timeout(400)
        check(not errors, f"χωρίς σφάλματα κονσόλας ({len(errors)}) {errors[:3]}")
        ctx.close()

        # 3) Κινητό: συρτάρι αναζήτησης
        print("\n[αλληλεπίδραση · κινητό]")
        errors = []
        ctx, page = new_page(browser, 390, 844, "dark", errors)
        page.goto(base + "/#/")
        wait_ready(page)
        page.click("#open-sidebar")
        page.wait_for_timeout(350)
        check(page.evaluate("document.activeElement.id") == "sb-search", "το συρτάρι ανοίγει με εστίαση στην αναζήτηση")
        page.keyboard.type("ΔΕΗ")
        page.wait_for_timeout(300)
        ok, sw = no_overflow(page)
        check(ok, f"συρτάρι: χωρίς οριζόντια κύλιση {sw}")
        shot(page, "drawer-mobile-dark", full=False, shots=shots)
        page.keyboard.press("Enter")
        page.wait_for_timeout(450)
        check(page.evaluate("location.hash") == "#/stock/PPC.AT", "«ΔΕΗ» + Enter → PPC.AT")
        check(not page.evaluate("document.getElementById('sidebar').classList.contains('is-open')"), "το συρτάρι κλείνει μετά την επιλογή")
        wait_ready(page)
        page.click("#open-sidebar")
        page.wait_for_timeout(300)
        page.click('.sb-tabs [data-tab="fav"]')
        page.wait_for_timeout(200)
        shot(page, "drawer-favorites-mobile-light", full=False, shots=False)
        page.keyboard.press("Escape")
        page.wait_for_timeout(300)
        check(not page.evaluate("document.getElementById('sidebar').classList.contains('is-open')"), "Esc κλείνει το συρτάρι")
        for w in (360,):
            page.set_viewport_size({"width": w, "height": 780})
            for name, route in ROUTES:
                page.goto(base + "/" + route)
                wait_ready(page)
                ok, sw = no_overflow(page)
                check(ok, f"{w}px · {name}: χωρίς οριζόντια κύλιση {sw}")
        check(not errors, f"χωρίς σφάλματα κονσόλας ({len(errors)}) {errors[:3]}")
        ctx.close()

        # 3β) Tablet / ενδιάμεσα πλάτη, ειδικές σελίδες μετοχών, μειωμένη κίνηση
        print("\n[ενδιάμεσα πλάτη & ειδικές σελίδες]")
        errors = []
        ctx, page = new_page(browser, 1024, 768, "dark", errors)
        for w in (760, 900, 1024, 1100, 1280):
            page.set_viewport_size({"width": w, "height": 800})
            for name, route in ROUTES:
                page.goto(base + "/" + route)
                wait_ready(page)
                ok, sw = no_overflow(page)
                if not ok:
                    check(False, f"{w}px · {name}: οριζόντια κύλιση {sw}")
            check(True, f"{w}px: έλεγχος υπερχείλισης σε όλες τις σελίδες")
        page.set_viewport_size({"width": 1024, "height": 768})
        shot(page, "settings-tablet-dark", full=False, shots=shots)
        page.set_viewport_size({"width": 1440, "height": 900})
        page.goto(base + "/#/stock/EYAPS.AT")
        wait_ready(page)
        check(page.locator(".decision--none").count() == 1, "μετοχή χωρίς δεδομένα → «ΧΩΡΙΣ ΣΗΜΑ» και κενή κατάσταση")
        shot(page, "stock-nodata-desktop-dark", full=False, shots=shots)
        page.goto(base + "/#/stock/ZZZ.AT")
        wait_ready(page)
        check(page.locator("[data-add='ZZZ.AT']").count() == 1, "άγνωστη μετοχή → «δεν βρέθηκε» με κουμπί προσθήκης")
        check(not [e for e in errors if "404" not in e], f"χωρίς σφάλματα κονσόλας εκτός του αναμενόμενου 404 {errors[:3]}")
        ctx.close()
        rctx = browser.new_context(viewport={"width": 1440, "height": 900}, reduced_motion="reduce")
        rp = rctx.new_page()
        rp.goto(base + "/#/stock/OPAP.AT")
        wait_ready(rp)
        anim = rp.evaluate("getComputedStyle(document.querySelector('.orb-glow')).animationName")
        check(anim == "none", f"prefers-reduced-motion → χωρίς «ανάσα» της σφαίρας ({anim})")
        rctx.close()

        # 4) Χωρίς δεδομένα (onboarding) και διακομιστής εκτός
        print("\n[ειδικές καταστάσεις]")
        errors = []
        ctx, page = new_page(browser, 1440, 900, "dark", errors)
        page.goto(base_empty + "/#/")
        wait_ready(page)
        check(page.locator(".onboard [data-job='refresh']").count() == 1 and page.locator(".onboard [data-job='demo']").count() == 1,
              "mode=empty → κάρτα έναρξης με δύο κουμπιά")
        shot(page, "empty-desktop-dark", full=False, shots=shots)
        page.goto(base_down + "/#/")
        page.wait_for_selector(".fatal", timeout=8000)
        check(page.locator(".fatal .code").first.inner_text().startswith("python -m athex_orb serve"), "διακομιστής εκτός → οδηγίες εκκίνησης")
        shot(page, "offline-desktop-dark", full=False, shots=shots)
        ctx.close()
        ctx, page = new_page(browser, 390, 844, "light", [])
        page.goto(base_down + "/#/")
        page.wait_for_selector(".fatal", timeout=8000)
        ok, sw = no_overflow(page)
        check(ok, f"εκτός · κινητό: χωρίς οριζόντια κύλιση {sw}")
        shot(page, "offline-mobile-light", full=False, shots=shots)
        ctx.close()
        ctx, page = new_page(browser, 390, 844, "light", [])
        page.goto(base_empty + "/#/")
        wait_ready(page)
        ok, sw = no_overflow(page)
        check(ok, f"empty · κινητό: χωρίς οριζόντια κύλιση {sw}")
        shot(page, "empty-mobile-light", full=False, shots=shots)
        ctx.close()
        ctx, page = new_page(browser, 1440, 900, "dark", [])
        page.goto(base_empty + "/#/")
        wait_ready(page)
        page.click(".onboard [data-job='demo']")
        page.wait_for_selector(".kpis", timeout=20000)
        check(page.locator(".status-pill.is-demo").count() == 1, "«Δοκιμή με δεδομένα επίδειξης» → δεδομένα & ένδειξη ΕΠΙΔΕΙΞΗ")
        ctx.close()
        # σφάλματα 503 εμφανίζονται ως αναμενόμενα στην κονσόλα του browser· δεν μετράνε εδώ
        browser.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-shots", action="store_true")
    args = ap.parse_args()
    p_empty, p_down = free_port(), free_port()
    procs = [start_mock(args.port), start_mock(p_empty, "--mode", "empty", reuse=False),
             start_mock(p_down, "--api-down", reuse=False)]
    try:
        run(f"http://127.0.0.1:{args.port}", f"http://127.0.0.1:{p_empty}", f"http://127.0.0.1:{p_down}",
            shots=not args.no_shots)
    finally:
        for pr in procs:
            if pr:
                pr.terminate()
    print(f"\n{'ΟΛΑ ΚΑΛΑ' if not failures else f'{len(failures)} ΑΠΟΤΥΧΙΕΣ'}")
    for f in failures:
        print("  ✕", f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
