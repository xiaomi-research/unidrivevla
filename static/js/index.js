const io = new IntersectionObserver((entries) => {
  entries.forEach((e) => {
    const v = e.target;
    if (e.isIntersecting) {
      if (v.dataset.src && v.getAttribute("src") !== v.dataset.src) v.src = v.dataset.src;
      v.play().catch(() => {});
    } else {
      v.pause();
    }
  });
}, { rootMargin: "200px 0px" });

document.querySelectorAll("video[data-src]").forEach((v) => io.observe(v));

const copyBtn = document.getElementById("copy-bib");
copyBtn?.addEventListener("click", async () => {
  const text = document.getElementById("bibtex-text")?.innerText || "";
  await navigator.clipboard.writeText(text);
  copyBtn.textContent = "Copied";
  setTimeout(() => { copyBtn.textContent = "Copy"; }, 1200);
});
