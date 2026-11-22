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
