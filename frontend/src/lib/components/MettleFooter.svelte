<script lang="ts">
  const year = new Date().getFullYear();
</script>

<footer class="mettle-footer">
  <div class="container">
    <div class="footer-content">
      <div class="footer-brand">
        <span class="footer-logo" aria-hidden="true">
          <img src="/favicon.svg" alt="" width="28" height="28" />
        </span>
        <div>
          <p class="footer-title">METTLE</p>
          <p class="footer-tagline">
            Machine Evaluation Through Turing-inverse Logic Examination
          </p>
          <p class="footer-mission">
            Proving agents are real, free, genuine, safe, and capable of novel
            reasoning.
          </p>
        </div>
      </div>

      <div class="footer-links">
        <div class="footer-section">
          <h3>Verification</h3>
          <a href="/#suites">11 Suites</a>
          <a href="/#how-it-works">How It Works</a>
          <a href="/docs">Docs</a>
        </div>
        <div class="footer-section">
          <h3>Ecosystem</h3>
          <a
            href="https://creed.space"
            target="_blank"
            rel="noopener noreferrer"
            aria-label="Creed Space (opens in new tab)"
          >
            Creed Space <i
              class="fa-solid fa-arrow-up-right-from-square external-icon"
              aria-hidden="true"
            ></i>
          </a>
          <a
            href="https://millos.net"
            target="_blank"
            rel="noopener noreferrer"
            aria-label="MillOS (opens in new tab)"
          >
            MillOS <i
              class="fa-solid fa-arrow-up-right-from-square external-icon"
              aria-hidden="true"
            ></i>
          </a>
        </div>
        <div class="footer-section">
          <h3>Open Source</h3>
          <a
            href="https://github.com/Creed-Space"
            target="_blank"
            rel="noopener noreferrer"
            aria-label="GitHub (opens in new tab)"
          >
            GitHub <i
              class="fa-solid fa-arrow-up-right-from-square external-icon"
              aria-hidden="true"
            ></i>
          </a>
          <a
            href="https://github.com/Creed-Space"
            target="_blank"
            rel="noopener noreferrer"
            aria-label="Contribute on GitHub (opens in new tab)"
          >
            Contribute <i
              class="fa-solid fa-arrow-up-right-from-square external-icon"
              aria-hidden="true"
            ></i>
          </a>
          <span class="footer-license-badge">
            <i class="fa-solid fa-scale-balanced" aria-hidden="true"></i>
            MIT Licensed
          </span>
        </div>
      </div>
    </div>

    <div class="footer-bottom">
      <p>
        &copy; {year} Creed Space &mdash; Code released under
        <a
          href="https://opensource.org/licenses/MIT"
          target="_blank"
          rel="noopener noreferrer"
          class="license-link">MIT License</a
        >
      </p>
      <a
        href="https://creed.space"
        target="_blank"
        rel="noopener noreferrer"
        class="footer-built-by"
      >
        <img
          src="/brand/mark.svg"
          alt=""
          width="18"
          height="18"
          class="creed-mark"
          aria-hidden="true"
        />
        Built by Creed Space
      </a>
    </div>
  </div>
</footer>

<style>
  .mettle-footer {
    background: linear-gradient(
      180deg,
      color-mix(in srgb, var(--color-bg-elevated) 92%, transparent) 0%,
      color-mix(in srgb, var(--color-bg) 96%, transparent) 100%
    );
    border-top: 1px solid color-mix(in srgb, var(--color-teal) 10%, transparent);
    padding: var(--space-xl) 0 var(--space-lg);
    margin-top: 0;
    -webkit-backdrop-filter: blur(16px);
    backdrop-filter: blur(16px);
    position: relative;
  }

  .mettle-footer::before {
    content: "";
    position: absolute;
    top: 0;
    left: 50%;
    transform: translateX(-50%);
    width: 400px;
    height: 2px;
    background: linear-gradient(
      90deg,
      transparent,
      color-mix(in srgb, var(--color-teal) 60%, transparent),
      color-mix(in srgb, var(--color-teal-light) 42%, transparent),
      transparent
    );
  }

  .footer-content {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: var(--space-xl);
    margin-bottom: var(--space-xl);
  }

  .footer-brand {
    display: flex;
    align-items: flex-start;
    gap: var(--space-md);
  }

  .footer-logo {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 44px;
    height: 44px;
    border-radius: 12px;
    background: linear-gradient(
      135deg,
      color-mix(in srgb, var(--color-teal) 18%, transparent),
      color-mix(in srgb, var(--color-teal-light) 10%, transparent)
    );
    color: var(--color-teal);
    font-size: 1.25rem;
    flex-shrink: 0;
  }

  /* Wordmark: solid caps at 0.24em tracking, single ink (Creed family). */
  .footer-title {
    font-weight: 600;
    font-size: 0.95rem;
    letter-spacing: 0.24em;
    margin-bottom: var(--space-xs);
    color: var(--color-teal);
  }


  .footer-tagline {
    color: var(--color-text-muted);
    font-size: 0.8125rem;
    max-width: 280px;
  }

  .footer-mission {
    color: var(--color-text-subtle, rgba(255, 255, 255, 0.4));
    font-size: 0.75rem;
    max-width: 280px;
    margin-top: var(--space-xs);
    line-height: 1.4;
  }

  .footer-links {
    display: flex;
    gap: var(--space-2xl);
  }

  .footer-section h3 {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--color-text-muted);
    margin-bottom: var(--space-sm);
  }

  .footer-section a {
    display: block;
    color: var(--color-text);
    text-decoration: none;
    font-size: 0.875rem;
    padding: var(--space-xs) 0;
    transition: color var(--transition-fast);
  }

  .footer-section a:hover {
    color: var(--color-teal);
    text-decoration: none;
  }

  .footer-section a:focus-visible {
    outline: 0;
    box-shadow: var(--focus-ring);
  }

  .external-icon {
    font-size: 0.75rem;
    opacity: 0.5;
  }

  .footer-bottom {
    padding-top: var(--space-lg);
    border-top: 1px solid rgba(255, 255, 255, 0.06);
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.8125rem;
    color: var(--color-text-muted);
  }

  .license-link {
    color: var(--color-teal-strong);
    text-decoration: none;
    font-weight: 600;
  }

  .license-link:hover {
    text-decoration: underline;
  }

  .footer-license-badge {
    display: inline-flex;
    align-items: center;
    gap: var(--space-xs);
    font-size: 0.75rem;
    color: var(--color-text-subtle, rgba(255, 255, 255, 0.4));
    padding: var(--space-xs) 0;
  }

  .footer-license-badge i {
    color: color-mix(in srgb, var(--color-teal) 58%, transparent);
    font-size: 0.625rem;
  }

  .footer-built-by {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--color-teal-strong);
    text-decoration: none;
    transition: opacity var(--transition-fast);
  }

  .footer-built-by:hover {
    opacity: 0.8;
    text-decoration: none;
  }

  .creed-mark {
    display: block;
    flex-shrink: 0;
  }

  @media (max-width: 768px) {
    .footer-content {
      flex-direction: column;
      gap: var(--space-xl);
    }

    .footer-links {
      width: 100%;
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: var(--space-lg);
    }

    .footer-bottom {
      flex-direction: column;
      gap: var(--space-sm);
      text-align: center;
    }

    .footer-brand {
      text-align: center;
      flex-direction: column;
      align-items: center;
    }

    .footer-tagline,
    .footer-mission {
      text-align: center;
    }
  }

  @media (max-width: 480px) {
    .footer-links {
      grid-template-columns: 1fr;
      text-align: center;
    }

    .footer-section a {
      display: inline-block;
    }
  }
</style>
