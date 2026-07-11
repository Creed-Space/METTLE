<script lang="ts">
  import { page } from "$app/stores";

  let mobileMenuOpen = $state(false);
  let mobileNavElement: HTMLElement | null = $state(null);
  let scrolled = $state(false);
  let theme = $state<"light" | "dark">("dark");
  let themeReady = $state(false);

  const currentPath = $derived($page.url.pathname);

  function isActive(href: string): boolean {
    if (href === "/") return currentPath === "/";
    return currentPath === href || currentPath.startsWith(href + "/");
  }

  // Track scroll position for header border effect
  $effect(() => {
    function handleScroll() {
      scrolled = window.scrollY > 10;
    }

    window.addEventListener("scroll", handleScroll, { passive: true });
    handleScroll();
    return () => window.removeEventListener("scroll", handleScroll);
  });

  $effect(() => {
    if (themeReady || typeof window === "undefined") return;
    const stored = window.localStorage.getItem("mettle-theme");
    if (stored === "light" || stored === "dark") {
      theme = stored;
    } else {
      theme = window.matchMedia?.("(prefers-color-scheme: dark)").matches
        ? "dark"
        : "light";
    }
    themeReady = true;
  });

  $effect(() => {
    if (!themeReady || typeof document === "undefined") return;
    document.documentElement.dataset.theme = theme;
    document
      .querySelector('meta[name="theme-color"]')
      ?.setAttribute("content", theme === "dark" ? "#061815" : "#14b8a6");
    try {
      window.localStorage.setItem("mettle-theme", theme);
    } catch {
      // Theme still updates for this session if storage is unavailable.
    }
  });

  // Focus trap for mobile menu
  $effect(() => {
    if (!mobileMenuOpen || !mobileNavElement) return;

    const focusable = mobileNavElement.querySelectorAll<HTMLElement>(
      'a[href], button, [tabindex]:not([tabindex="-1"])',
    );
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    first?.focus();

    function handleKeydown(e: KeyboardEvent) {
      if (e.key === "Escape") {
        mobileMenuOpen = false;
        return;
      }
      if (e.key !== "Tab") return;
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last?.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first?.focus();
      }
    }

    document.addEventListener("keydown", handleKeydown);
    return () => document.removeEventListener("keydown", handleKeydown);
  });
</script>

<header class="mettle-header" class:scrolled>
  <div class="container flex items-center justify-between">
    <a href="/" class="mettle-logo" aria-label="METTLE Home">
      <span class="mettle-logo-icon" aria-hidden="true">
        <img src="/favicon.svg" alt="" width="28" height="28" />
      </span>
      <span class="mettle-logo-text">
        METTLE
      </span>
    </a>

    <!-- Desktop Nav -->
    <nav class="mettle-nav desktop-nav" aria-label="Main navigation">
      <a href="/#suites" class="nav-link">Suites</a>
      <a
        href="/docs"
        class="nav-link"
        class:active={isActive("/docs")}
        aria-current={isActive("/docs") ? "page" : undefined}>Docs</a
      >
      <span class="nav-divider" aria-hidden="true"></span>
      <a
        href="https://creed.space"
        target="_blank"
        rel="noopener noreferrer"
        class="nav-link nav-link-brand"
        aria-label="Creed Space (opens in new tab)"
      >
        Creed Space
      </a>
    </nav>

    <button
      class="theme-toggle"
      type="button"
      aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
      onclick={() => (theme = theme === "dark" ? "light" : "dark")}
    >
      <i
        class={`fa-solid ${theme === "dark" ? "fa-sun" : "fa-moon"}`}
        aria-hidden="true"
      ></i>
      <span>{theme === "dark" ? "Light" : "Dark"}</span>
    </button>

    <!-- Mobile Menu Button -->
    <button
      class="mobile-menu-btn"
      onclick={() => (mobileMenuOpen = !mobileMenuOpen)}
      aria-expanded={mobileMenuOpen}
      aria-controls="mettle-mobile-nav"
      aria-label={mobileMenuOpen ? "Close menu" : "Open menu"}
    >
      <span class="hamburger" class:open={mobileMenuOpen}>
        <span></span>
        <span></span>
        <span></span>
      </span>
    </button>
  </div>

  <!-- Mobile Nav -->
  {#if mobileMenuOpen}
    <nav
      id="mettle-mobile-nav"
      class="mobile-nav"
      aria-label="Mobile navigation"
      bind:this={mobileNavElement}
    >
      <a
        href="/#suites"
        class="mobile-nav-link"
        onclick={() => (mobileMenuOpen = false)}>Suites</a
      >
      <a
        href="/docs"
        class="mobile-nav-link"
        aria-current={isActive("/docs") ? "page" : undefined}
        onclick={() => (mobileMenuOpen = false)}>Docs</a
      >
      <button
        class="mobile-theme-toggle"
        type="button"
        aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
        onclick={() => (theme = theme === "dark" ? "light" : "dark")}
      >
        <i
          class={`fa-solid ${theme === "dark" ? "fa-sun" : "fa-moon"}`}
          aria-hidden="true"
        ></i>
        <span>{theme === "dark" ? "Light theme" : "Dark theme"}</span>
      </button>
      <hr class="mobile-nav-divider" />
      <a
        href="https://creed.space"
        target="_blank"
        rel="noopener noreferrer"
        class="mobile-nav-link mobile-nav-brand"
        onclick={() => (mobileMenuOpen = false)}
      >
        Creed Space
      </a>
    </nav>
  {/if}
</header>

<style>
  .mettle-header {
    background: color-mix(in srgb, var(--color-bg-elevated) 86%, transparent);
    border-bottom: 1px solid
      color-mix(in srgb, var(--color-teal) 8%, transparent);
    padding: var(--space-sm) 0;
    position: sticky;
    top: 0;
    z-index: 100;
    -webkit-backdrop-filter: blur(20px) saturate(1.5);
    backdrop-filter: blur(20px) saturate(1.5);
    transition:
      border-color 0.3s ease,
      background 0.3s ease;
  }

  .mettle-header.scrolled {
    border-bottom: 1px solid
      color-mix(in srgb, var(--color-teal) 24%, transparent);
    background: color-mix(in srgb, var(--color-bg-elevated) 96%, transparent);
  }

  /* Logo */
  .mettle-logo {
    display: flex;
    align-items: center;
    gap: var(--space-sm);
    text-decoration: none;
    color: var(--color-text);
  }

  .mettle-logo:hover {
    text-decoration: none;
  }

  .mettle-logo-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 40px;
    height: 40px;
    border-radius: 10px;
    background: linear-gradient(
      135deg,
      color-mix(in srgb, var(--color-teal) 18%, transparent),
      color-mix(in srgb, var(--color-teal-light) 10%, transparent)
    );
    color: var(--color-teal);
    font-size: 1.125rem;
    transition: all 0.3s ease;
  }

  .mettle-logo:hover .mettle-logo-icon {
    background: linear-gradient(
      135deg,
      color-mix(in srgb, var(--color-teal) 26%, transparent),
      color-mix(in srgb, var(--color-teal-light) 14%, transparent)
    );
    box-shadow: 0 0 20px color-mix(in srgb, var(--color-teal) 16%, transparent);
  }

  /* Wordmark: solid caps at 0.24em tracking, single ink (Creed family). */
  .mettle-logo-text {
    font-weight: 600;
    font-size: 1.05rem;
    letter-spacing: 0.24em;
    color: var(--color-teal);
    white-space: nowrap;
  }

  .mettle-logo:hover .mettle-logo-text {
    color: var(--color-teal-light);
  }

  /* Desktop Nav */
  .desktop-nav {
    display: flex;
    align-items: center;
    gap: var(--space-lg);
  }

  .nav-link {
    color: var(--color-text-muted);
    text-decoration: none;
    font-size: 0.875rem;
    font-weight: 500;
    transition: color var(--transition-fast);
    padding: var(--space-xs) var(--space-sm);
    border-radius: var(--radius-sm);
  }

  .nav-link:hover {
    color: var(--color-text);
    text-decoration: none;
    background: color-mix(in srgb, var(--color-teal) 10%, transparent);
  }

  .nav-link:focus-visible {
    outline: 0;
    box-shadow: var(--focus-ring);
  }

  .nav-link.active {
    color: var(--color-teal);
  }

  .nav-divider {
    width: 1px;
    height: 16px;
    background: rgba(255, 255, 255, 0.15);
  }

  .nav-link-brand {
    display: flex;
    align-items: center;
    gap: var(--space-xs);
    color: var(--color-teal-strong);
  }

  .nav-link-brand:hover {
    color: var(--color-teal);
  }

  .theme-toggle,
  .mobile-theme-toggle {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: var(--space-xs);
    border: 1px solid var(--color-border);
    border-radius: 999px;
    background: var(--glass-bg);
    color: var(--color-text);
    font: inherit;
    font-size: 0.8125rem;
    font-weight: 700;
    min-height: 2.4rem;
    padding: var(--space-xs) var(--space-md);
    cursor: pointer;
  }

  .theme-toggle:hover,
  .mobile-theme-toggle:hover {
    border-color: var(--color-teal);
    background: var(--glass-bg-hover);
  }

  /* Mobile Menu Button */
  .mobile-menu-btn {
    display: none;
    background: none;
    border: none;
    padding: var(--space-sm);
    cursor: pointer;
    border-radius: var(--radius-sm);
  }

  .mobile-menu-btn:hover {
    background: rgba(255, 255, 255, 0.05);
  }
  .mobile-menu-btn:focus-visible {
    outline: 0;
    box-shadow: var(--focus-ring);
  }

  .hamburger {
    display: flex;
    flex-direction: column;
    gap: 5px;
    width: 22px;
    height: 18px;
    position: relative;
  }

  .hamburger span {
    display: block;
    height: 2px;
    width: 100%;
    background: var(--color-text);
    border-radius: 2px;
    transition: all 0.3s cubic-bezier(0.68, -0.55, 0.265, 1.55);
    position: absolute;
    left: 0;
  }

  .hamburger span:nth-child(1) {
    top: 0;
  }
  .hamburger span:nth-child(2) {
    top: 8px;
  }
  .hamburger span:nth-child(3) {
    top: 16px;
  }
  .hamburger.open span:nth-child(1) {
    transform: rotate(45deg);
    top: 8px;
  }
  .hamburger.open span:nth-child(2) {
    opacity: 0;
    transform: translateX(-10px);
  }
  .hamburger.open span:nth-child(3) {
    transform: rotate(-45deg);
    top: 8px;
  }

  /* Mobile Nav */
  .mobile-nav {
    display: none;
    flex-direction: column;
    padding: var(--space-md);
    background: color-mix(in srgb, var(--color-bg-elevated) 96%, transparent);
    border-top: 1px solid color-mix(in srgb, var(--color-teal) 10%, transparent);
    animation: slideDown 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    -webkit-backdrop-filter: blur(20px);
    backdrop-filter: blur(20px);
  }

  @keyframes slideDown {
    from {
      opacity: 0;
      transform: translateY(-10px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  .mobile-nav-link {
    display: flex;
    align-items: center;
    gap: var(--space-sm);
    padding: var(--space-md);
    color: var(--color-text);
    text-decoration: none;
    font-weight: 500;
    border-radius: var(--radius-md);
    transition: background var(--transition-fast);
  }

  .mobile-nav-link:hover {
    background: rgba(255, 255, 255, 0.05);
    text-decoration: none;
  }
  .mobile-nav-link:focus-visible,
  .mobile-theme-toggle:focus-visible {
    outline: 0;
    box-shadow: var(--focus-ring);
  }
  .mobile-nav-brand {
    color: var(--color-teal-strong);
  }
  .mobile-nav-divider {
    border: none;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
    margin: var(--space-sm) 0;
  }

  @media (max-width: 768px) {
    .desktop-nav {
      display: none;
    }
    .theme-toggle {
      display: none;
    }
    .mobile-menu-btn {
      display: block;
    }
    .mobile-nav {
      display: flex;
    }
  }

  @media (prefers-reduced-motion: reduce) {
    .mettle-header,
    .mettle-logo-icon,
    .nav-link,
    .mobile-nav-link,
    .hamburger span {
      transition: none;
    }

    .mobile-nav {
      animation: none;
    }
  }
</style>
