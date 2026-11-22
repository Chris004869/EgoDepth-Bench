"use strict";

// The full page and citation remain readable without JavaScript.
const copyButton = document.getElementById("copy-citation");
const citation = document.getElementById("bibtex");
const copyStatus = document.getElementById("copy-status");

if (copyButton && citation && copyStatus) {
  copyButton.hidden = false;
  copyButton.addEventListener("click", async () => {
    const text = citation.textContent.trim();
    let copied = false;
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
        copied = true;
      }
    } catch (_) {
      // Local file previews and browsers may deny clipboard permissions.
    }
    if (!copied) {
      const field = document.createElement("textarea");
      field.value = text;
      field.setAttribute("aria-label", "Citation to copy");
      field.style.cssText = "position:fixed;left:-9999px;top:0;";
      document.body.appendChild(field);
      field.select();
      try {
        copied = document.execCommand("copy");
      } catch (_) {
        copied = false;
      }
      field.remove();
      copyButton.focus({ preventScroll: true });
    }
    copyStatus.textContent = copied
      ? "BibTeX copied to clipboard."
      : "Clipboard unavailable. Select the citation above and copy it manually.";
  });
}

// A single, aligned camera-2 frame supplies both renderings.
const viewer = document.querySelector("#viewer");
const modes = document.querySelector(".mode-controls");
const handle = viewer.querySelector(".wipe-input");
modes.hidden = false;
modes.addEventListener("click", event => {
  const button = event.target.closest("[data-layout]");
  if (!button) return;
  const wipe = button.dataset.layout === "wipe";
  viewer.classList.toggle("is-wipe", wipe);
  handle.hidden = !wipe;
  for (const item of modes.querySelectorAll("button")) {
    item.setAttribute("aria-pressed", String(item === button));
  }
});
handle.addEventListener("input", () => viewer.style.setProperty("--split", handle.value + "%"));

// Printed pages and visitors without JavaScript keep all scenes and tables.
for (const group of document.querySelectorAll("[data-switches]")) {
  const buttons = [...group.querySelectorAll("button")];
  function select(active) {
    for (const button of buttons) {
      const panel = document.getElementById(button.getAttribute("aria-controls"));
      panel.hidden = button !== active;
      button.setAttribute("aria-pressed", String(button === active));
    }
  }
  group.hidden = false;
  select(buttons[0]);
  group.addEventListener("click", event => {
    const button = event.target.closest("button");
    if (button) select(button);
  });
}

const video = document.getElementById("overview-video");
const jumps = document.querySelector(".chapter-jumps");
jumps.hidden = false;
jumps.addEventListener("click", event => {
  const button = event.target.closest("[data-time]");
  if (!button) return;
  video.currentTime = Number(button.dataset.time);
  video.play().catch(() => { video.focus(); });
});
