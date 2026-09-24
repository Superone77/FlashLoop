(() => {
  document.querySelectorAll('[data-copy-target]').forEach((button) => {
    const source = document.getElementById(button.dataset.copyTarget);
    if (!source) return;

    button.addEventListener('click', async () => {
      const text = source.textContent.trim();
      try {
        if (navigator.clipboard && window.isSecureContext) {
          await navigator.clipboard.writeText(text);
        } else {
          const field = document.createElement('textarea');
          field.value = text;
          field.style.position = 'fixed';
          field.style.opacity = '0';
          document.body.append(field);
          field.select();
          if (!document.execCommand('copy')) throw new Error('Copy failed');
          field.remove();
        }
        button.textContent = 'Copied';
      } catch {
        button.textContent = 'Select text';
        const selection = window.getSelection();
        const range = document.createRange();
        range.selectNodeContents(source);
        selection.removeAllRanges();
        selection.addRange(range);
      }
      window.setTimeout(() => { button.textContent = 'Copy'; }, 2200);
    });
  });
})();
