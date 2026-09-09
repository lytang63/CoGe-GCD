function setupCopyButton() {
  const button = document.querySelector('.copy-button');
  const citation = document.getElementById('bibtex');
  const status = document.querySelector('.copy-status');
  if (!button || !citation || !status) return;

  button.addEventListener('click', async () => {
    const text = citation.textContent.trim();
    try {
      await navigator.clipboard.writeText(text);
    } catch (error) {
      const textarea = document.createElement('textarea');
      textarea.value = text;
      textarea.style.position = 'fixed';
      textarea.style.opacity = '0';
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand('copy');
      textarea.remove();
    }
    button.textContent = button.dataset.copiedLabel || 'Copied';
    status.textContent = status.dataset.copiedMessage || 'BibTeX copied to clipboard.';
    window.setTimeout(() => {
      button.textContent = button.dataset.defaultLabel || 'Copy BibTeX';
      status.textContent = '';
    }, 1800);
  });
}

function setupReveal() {
  const sections = document.querySelectorAll('main > section');
  if (!('IntersectionObserver' in window)) {
    sections.forEach((section) => section.classList.add('is-visible'));
    return;
  }
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add('is-visible');
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.08, rootMargin: '0px 0px -40px' });
  sections.forEach((section) => {
    section.classList.add('reveal-ready');
    observer.observe(section);
  });
}

function setupHeroGlow() {
  const hero = document.querySelector('.hero-block');
  if (!hero || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  hero.addEventListener('pointermove', (event) => {
    const rect = hero.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * 100;
    const y = ((event.clientY - rect.top) / rect.height) * 100;
    hero.style.setProperty('--pointer-x', x + '%');
    hero.style.setProperty('--pointer-y', y + '%');
  });
}

setupCopyButton();
setupReveal();
setupHeroGlow();

