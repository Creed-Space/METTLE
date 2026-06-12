<script lang="ts">
  /**
   * ParticleField — Canvas-based constellation/starfield effect.
   * Tiny dots drift slowly and connect with lines when near each other,
   * forming an organic network. Great as a section background.
   */
  interface Props {
    /** Number of particles */
    count?: number;
    /** Connection distance in px */
    connectionDistance?: number;
    /** Particle color */
    color?: string;
    /** Line color */
    lineColor?: string;
    /** Max particle speed */
    speed?: number;
    /** Particle size range [min, max] */
    sizeRange?: [number, number];
  }

  let {
    count = 50,
    connectionDistance = 120,
    color = "rgba(20, 184, 166, 0.4)",
    lineColor = "rgba(20, 184, 166, 0.1)",
    speed = 0.3,
    sizeRange = [1, 2.5] as [number, number],
  }: Props = $props();

  let canvas: HTMLCanvasElement | null = $state(null);

  interface Particle {
    x: number;
    y: number;
    vx: number;
    vy: number;
    size: number;
  }

  $effect(() => {
    if (!canvas || typeof window === "undefined") return;

    const context = canvas.getContext("2d");
    if (!context) return;
    const ctx: CanvasRenderingContext2D = context;

    let animId: number = 0;
    let particles: Particle[] = [];
    let width = 0;
    let height = 0;
    let visible = true;
    let prefersReducedMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;
    const motionQuery = window.matchMedia("(prefers-reduced-motion: reduce)");

    function resize() {
      if (!canvas) return;
      const parent = canvas.parentElement;
      if (!parent) return;
      width = parent.clientWidth;
      height = parent.clientHeight;
      const pixelRatio = window.devicePixelRatio || 1;
      canvas.width = width * pixelRatio;
      canvas.height = height * pixelRatio;
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      ctx.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    }

    function initParticles() {
      particles = Array.from({ length: count }, () => ({
        x: Math.random() * width,
        y: Math.random() * height,
        vx: (Math.random() - 0.5) * speed * 2,
        vy: (Math.random() - 0.5) * speed * 2,
        size: sizeRange[0] + Math.random() * (sizeRange[1] - sizeRange[0]),
      }));
    }

    function draw() {
      ctx.clearRect(0, 0, width, height);

      // Draw connections
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i]!.x - particles[j]!.x;
          const dy = particles[i]!.y - particles[j]!.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < connectionDistance) {
            const opacity = 1 - dist / connectionDistance;
            ctx.beginPath();
            ctx.strokeStyle = lineColor.replace(
              /[\d.]+\)$/,
              `${opacity * 0.15})`,
            );
            ctx.lineWidth = 0.5;
            ctx.moveTo(particles[i]!.x, particles[i]!.y);
            ctx.lineTo(particles[j]!.x, particles[j]!.y);
            ctx.stroke();
          }
        }
      }

      // Draw and update particles
      for (const p of particles) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();

        if (!prefersReducedMotion) {
          p.x += p.vx;
          p.y += p.vy;

          // Wrap around edges
          if (p.x < -10) p.x = width + 10;
          if (p.x > width + 10) p.x = -10;
          if (p.y < -10) p.y = height + 10;
          if (p.y > height + 10) p.y = -10;
        }
      }

      if (!prefersReducedMotion && visible && !document.hidden) {
        animId = requestAnimationFrame(draw);
      }
    }

    function start() {
      cancelAnimationFrame(animId);
      if (visible && !document.hidden) {
        animId = requestAnimationFrame(draw);
      }
    }

    function handleResize() {
      resize();
      initParticles();
      start();
    }

    function handleVisibilityChange() {
      if (document.hidden) {
        cancelAnimationFrame(animId);
      } else {
        start();
      }
    }

    resize();
    initParticles();
    draw();

    const observer = new IntersectionObserver((entries) => {
      visible = entries.some((entry) => entry.isIntersecting);
      if (visible) {
        start();
      } else {
        cancelAnimationFrame(animId);
      }
    });
    observer.observe(canvas);

    function handleMotionChange(event: MediaQueryListEvent) {
      prefersReducedMotion = event.matches;
      start();
    }

    window.addEventListener("resize", handleResize, { passive: true });
    document.addEventListener("visibilitychange", handleVisibilityChange);
    motionQuery.addEventListener("change", handleMotionChange);

    return () => {
      cancelAnimationFrame(animId);
      observer.disconnect();
      window.removeEventListener("resize", handleResize);
      document.removeEventListener("visibilitychange", handleVisibilityChange);
      motionQuery.removeEventListener("change", handleMotionChange);
    };
  });
</script>

<div class="particle-field" aria-hidden="true">
  <canvas bind:this={canvas}></canvas>
</div>

<style>
  .particle-field {
    position: absolute;
    inset: 0;
    overflow: hidden;
    pointer-events: none;
    z-index: -1;
  }

  canvas {
    display: block;
  }
</style>
