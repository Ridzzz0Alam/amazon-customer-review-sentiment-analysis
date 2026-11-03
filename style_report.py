"""Adds the light and dark theme and the page layout to the exported report.

    jupyter nbconvert --to html sentiment_analysis.ipynb --output-dir reports
    python style_report.py

The theme lives in reports/theme/. It is inlined so the report stays one file.
Running the script again replaces the earlier styling, so it is safe to repeat.
"""

import re
from pathlib import Path

REPORT = Path("reports/sentiment_analysis.html")
THEME_DIR = Path("reports/theme")
TITLE = "Amazon Review Sentiment"

FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com"/>'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400'
    '&family=JetBrains+Mono:wght@400&display=swap"/>'
)

# Runs before the page paints so a saved or system dark theme never flashes light.
EARLY_THEME = """<script>
(function () {
  var theme = null;
  try { theme = localStorage.getItem("rt-theme"); } catch (e) {}
  if (theme !== "light" && theme !== "dark") {
    theme = window.matchMedia && matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  document.documentElement.dataset.theme = theme;
})();
</script>"""


def block(name, content):
    return f"<!-- report-theme:{name} -->\n{content}\n<!-- /report-theme:{name} -->"


def strip_blocks(html):
    return re.sub(r"<!-- report-theme:(\w+) -->.*?<!-- /report-theme:\1 -->\n?", "", html, flags=re.S)


def main():
    html = strip_blocks(REPORT.read_text(encoding="utf-8"))
    css = (THEME_DIR / "theme.css").read_text(encoding="utf-8")
    js = (THEME_DIR / "theme.js").read_text(encoding="utf-8")
    header = (THEME_DIR / "header.html").read_text(encoding="utf-8")
    footer = (THEME_DIR / "footer.html").read_text(encoding="utf-8")

    html = re.sub(r"<title>.*?</title>", f"<title>{TITLE}</title>", html, count=1, flags=re.S)
    head = block("head", f"{EARLY_THEME}\n{FONTS}\n<style>\n{css}</style>")
    html = html.replace("</head>", head + "\n</head>", 1)
    html = re.sub(r"(<body[^>]*>)", lambda m: m.group(1) + "\n" + block("header", header), html, count=1)
    html = html.replace("</body>", block("footer", f"{footer}<script>\n{js}</script>") + "\n</body>", 1)

    REPORT.write_text(html, encoding="utf-8")
    print(f"Styled {REPORT} ({REPORT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
