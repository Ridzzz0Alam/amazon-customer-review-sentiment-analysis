// Theme toggle, table of contents, code toggle and small reading aids for the report.
(function () {
  const root = document.documentElement;
  const body = document.body;

  function store(key, value) {
    try { localStorage.setItem(key, value); } catch (e) { /* storage blocked */ }
  }
  function load(key) {
    try { return localStorage.getItem(key); } catch (e) { return null; }
  }

  // Theme. The initial value is set in the head so the page never flashes.
  const themeBtn = document.getElementById("rt-theme-toggle");
  function applyTheme(theme) {
    root.dataset.theme = theme;
    body.dataset.jpThemeLight = String(theme === "light");
    themeBtn.setAttribute("aria-label", theme === "light" ? "Switch to dark theme" : "Switch to light theme");
  }
  applyTheme(root.dataset.theme || "light");
  themeBtn.addEventListener("click", () => {
    const next = root.dataset.theme === "dark" ? "light" : "dark";
    applyTheme(next);
    store("rt-theme", next);
  });

  // Show or hide every code cell.
  const codeBtn = document.getElementById("rt-code-toggle");
  function applyCode(hidden) {
    body.classList.toggle("rt-hide-code", hidden);
    codeBtn.setAttribute("aria-pressed", String(hidden));
    codeBtn.textContent = hidden ? "Show code" : "Hide code";
  }
  applyCode(load("rt-hide-code") === "1");
  codeBtn.addEventListener("click", () => {
    const hidden = !body.classList.contains("rt-hide-code");
    applyCode(hidden);
    store("rt-hide-code", hidden ? "1" : "0");
  });

  // Table of contents with the current section highlighted.
  const headings = [...document.querySelectorAll("main .jp-RenderedMarkdown h2[id], main .jp-RenderedMarkdown h3[id]")];
  const toc = document.querySelector(".rt-toc");
  const links = new Map();
  headings.forEach((h) => {
    const a = document.createElement("a");
    a.href = "#" + h.id;
    a.textContent = [...h.childNodes]
      .filter((n) => n.nodeType === Node.TEXT_NODE)
      .map((n) => n.textContent)
      .join("")
      .trim();
    if (h.tagName === "H3") a.className = "sub";
    // The CSS scrolls smoothly. This marks the heading once it is reached.
    a.addEventListener("click", () => {
      let done = false;
      const flash = () => {
        if (done) return;
        done = true;
        h.classList.remove("rt-flash");
        void h.offsetWidth;   // restart the animation on repeat clicks
        h.classList.add("rt-flash");
      };
      window.addEventListener("scrollend", flash, { once: true });
      setTimeout(flash, 1500);   // browsers without scrollend, or no scroll needed
    });
    toc.appendChild(a);
    links.set(h, a);
  });

  const topBtn = document.querySelector(".rt-top");
  let ticking = false;

  function onScroll() {
    ticking = false;
    topBtn.classList.toggle("show", window.scrollY > window.innerHeight);

    let current = null;
    for (const h of headings) {
      if (h.getBoundingClientRect().top < 120) current = h;
      else break;
    }
    links.forEach((a, h) => a.classList.toggle("active", h === current));
  }
  window.addEventListener("scroll", () => {
    if (!ticking) {
      ticking = true;
      requestAnimationFrame(onScroll);
    }
  }, { passive: true });
  onScroll();
  topBtn.addEventListener("click", () => window.scrollTo({ top: 0 }));

  // Copy button on each code cell.
  document.querySelectorAll(".jp-CodeCell .jp-Cell-inputWrapper").forEach((cell) => {
    const pre = cell.querySelector(".highlight pre");
    if (!pre) return;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "rt-copy";
    btn.textContent = "Copy";
    btn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(pre.innerText);
        btn.textContent = "Copied";
      } catch (e) {
        btn.textContent = "Copy failed";
      }
      setTimeout(() => { btn.textContent = "Copy"; }, 1500);
    });
    cell.appendChild(btn);
  });

  // Click a chart to see it full size.
  const box = document.querySelector(".rt-lightbox");
  const boxImg = box.querySelector("img");
  document.querySelectorAll(".jp-RenderedImage img").forEach((img) => {
    img.alt = "Chart from the analysis";
    img.addEventListener("click", () => {
      boxImg.src = img.src;
      box.classList.add("open");
    });
  });
  box.addEventListener("click", () => box.classList.remove("open"));
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") box.classList.remove("open");
  });
})();
