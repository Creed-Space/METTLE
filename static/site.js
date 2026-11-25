/**
 * METTLE - Shared Site Behavior
 * Header scroll state, accessible mobile menu, docs scroll-spy, and copy controls.
 */
(function () {
    'use strict';

    // Scroll-aware header (throttled via requestAnimationFrame)
    (function initHeaderScroll() {
        var header = document.getElementById('mettle-header');
        if (!header) return;
        var ticking = false;
        function syncHeader() {
            ticking = false;
            if (window.scrollY > 10) {
                header.classList.add('scrolled');
            } else {
                header.classList.remove('scrolled');
            }
        }
        window.addEventListener('scroll', function () {
            if (!ticking) {
                ticking = true;
                window.requestAnimationFrame(syncHeader);
            }
        }, { passive: true });
        syncHeader();
    })();

    // Accessible mobile menu toggle
    (function initMobileMenu() {
        var btn = document.getElementById('mobile-menu-btn');
        var nav = document.getElementById('mettle-mobile-nav');
        var hamburger = document.getElementById('hamburger');
        if (!btn || !nav) return;

        function closeMenu(returnFocus) {
            nav.hidden = true;
            btn.setAttribute('aria-expanded', 'false');
            btn.setAttribute('aria-label', 'Open menu');
            if (hamburger) hamburger.classList.remove('open');
            if (returnFocus) btn.focus();
        }

        btn.addEventListener('click', function () {
            var isOpen = nav.hidden === false;
            if (isOpen) {
                closeMenu(false);
            } else {
                nav.hidden = false;
                btn.setAttribute('aria-expanded', 'true');
                btn.setAttribute('aria-label', 'Close menu');
                if (hamburger) hamburger.classList.add('open');
            }
        });

        nav.querySelectorAll('a').forEach(function (link) {
            link.addEventListener('click', function () {
                closeMenu(false);
            });
        });

        document.addEventListener('keydown', function (e) {
            if (e.key === 'Escape' && !nav.hidden) {
                closeMenu(true);
            }
        });
    })();

    // Docs TOC scroll-spy with rAF throttling and aria-current="location"
    (function initDocsScrollSpy() {
        var tocItems = document.querySelectorAll('.toc-item');
        if (!tocItems.length) return;

        var sectionIds = [];
        tocItems.forEach(function (item) {
            var href = item.getAttribute('href');
            if (href && href.charAt(0) === '#') {
                sectionIds.push(href.slice(1));
            }
        });
        if (!sectionIds.length) return;

        var ticking = false;
        function updateActive() {
            ticking = false;
            var current = sectionIds[0];
            for (var i = sectionIds.length - 1; i >= 0; i--) {
                var el = document.getElementById(sectionIds[i]);
                if (el && el.getBoundingClientRect().top <= 160) {
                    current = sectionIds[i];
                    break;
                }
            }
            if ((window.innerHeight + window.scrollY) >= (document.body.scrollHeight - 240)) {
                current = sectionIds[sectionIds.length - 1];
            }

            tocItems.forEach(function (item) {
                var isCurrent = item.getAttribute('href') === '#' + current;
                item.classList.toggle('active', isCurrent);
                if (isCurrent) {
                    item.setAttribute('aria-current', 'location');
                } else {
                    item.removeAttribute('aria-current');
                }
            });
        }

        window.addEventListener('scroll', function () {
            if (!ticking) {
                ticking = true;
                window.requestAnimationFrame(updateActive);
            }
        }, { passive: true });
        updateActive();
    })();

    // Copy helper for code blocks on /guide
    (function initCodeCopyButtons() {
        var blocks = document.querySelectorAll('.code-block-new');
        if (!blocks.length) return;

        function copyText(text) {
            if (navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
                return navigator.clipboard.writeText(text);
            }
            return new Promise(function (resolve, reject) {
                var area = document.createElement('textarea');
                area.value = text;
                area.className = 'sr-only';
                area.setAttribute('readonly', '');
                document.body.appendChild(area);
                area.select();
                try {
                    document.execCommand('copy');
                    resolve();
                } catch (err) {
                    reject(err);
                } finally {
                    document.body.removeChild(area);
                }
            });
        }

        blocks.forEach(function (block) {
            if (block.parentElement && block.parentElement.classList.contains('code-block-wrapper')) {
                return;
            }
            var wrapper = document.createElement('div');
            wrapper.className = 'code-block-wrapper';
            block.parentNode.insertBefore(wrapper, block);
            wrapper.appendChild(block);

            var btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'copy-code-btn';
            btn.textContent = 'Copy';
            btn.setAttribute('aria-label', 'Copy code snippet');
            wrapper.appendChild(btn);

            var resetTimer = null;
            btn.addEventListener('click', function () {
                var codeEl = block.querySelector('code');
                var text = codeEl ? codeEl.textContent : block.textContent;
                copyText(text || '').then(function () {
                    btn.textContent = 'Copied';
                    btn.classList.add('copied');
                    if (resetTimer) window.clearTimeout(resetTimer);
                    resetTimer = window.setTimeout(function () {
                        btn.textContent = 'Copy';
                        btn.classList.remove('copied');
                    }, 2000);
                }).catch(function () {
                    btn.textContent = 'Copy failed';
                    if (resetTimer) window.clearTimeout(resetTimer);
                    resetTimer = window.setTimeout(function () {
                        btn.textContent = 'Copy';
                    }, 2000);
                });
            });
        });
    })();
})();
