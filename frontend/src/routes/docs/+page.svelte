<script lang="ts">
  /**
   * METTLE Documentation
   * Self-hosted verification + optional notarization service
   */
  import { Breadcrumb } from "$lib/components";

  const breadcrumbItems = [
    { label: "METTLE", href: "/", icon: "fa-fire-flame-curved" },
    { label: "Docs", icon: "fa-book" },
  ];

  const tocSections = [
    { id: "getting-started", label: "Getting Started", icon: "fa-rocket" },
    { id: "self-hosted", label: "Self-Hosted", icon: "fa-server" },
    { id: "credentials", label: "Credentials", icon: "fa-certificate" },
    { id: "notarization", label: "Notarization", icon: "fa-stamp" },
    { id: "notarize-api", label: "Notarize API", icon: "fa-globe" },
    { id: "challenge-types", label: "Challenges", icon: "fa-puzzle-piece" },
    { id: "scoring", label: "Scoring", icon: "fa-chart-bar" },
    { id: "rate-limits", label: "Rate Limits", icon: "fa-gauge-high" },
    { id: "sdk-examples", label: "SDKs", icon: "fa-code" },
  ];

  const challengeTypes = [
    {
      name: "Procedural reasoning",
      format: "Structured answer",
      detail:
        "Fresh math, graph, and logic prompts generated from the run seed.",
    },
    {
      name: "Calibration probes",
      format: "Probability + rationale",
      detail:
        "Confidence is scored against outcomes, so performed certainty is penalized.",
    },
    {
      name: "Self-reference checks",
      format: "Prediction + follow-up",
      detail:
        "The verifier asks the agent to model its own next-step behavior, then measures drift.",
    },
    {
      name: "Governance gates",
      format: "Policy-bound decision",
      detail:
        "Action refusal, constitutional binding, and override resistance must stay coherent.",
    },
  ];

  const scoringRules = [
    {
      label: "Suite score",
      value: "0.0–1.0",
      detail: "Weighted challenge results within one suite.",
    },
    {
      label: "Suite pass",
      value: "0.70+",
      detail: "Minimum score required for that suite to count.",
    },
    {
      label: "Full pass",
      value: "0.80+",
      detail: "Aggregate target for full verification mode.",
    },
    {
      label: "Unsafe override",
      value: "Fail closed",
      detail:
        "Failed harm-refusal or provenance checks override the aggregate score.",
    },
  ];

  const rateLimits = [
    {
      endpoint: "POST /notarize/seed",
      limit: "60/min",
      notes: "Seed generation for deterministic local runs.",
    },
    {
      endpoint: "POST /notarize",
      limit: "30/min",
      notes: "Credential validation and Ed25519 signing.",
    },
    {
      endpoint: "GET /.well-known/jwks.json",
      limit: "300/min",
      notes: "Public key discovery for verifiers.",
    },
  ];

  function focusScrollable(node: HTMLElement) {
    node.tabIndex = 0;

    return {
      destroy() {
        node.removeAttribute("tabindex");
      },
    };
  }

  let activeSection = $state("getting-started");

  function scrollToSection(id: string) {
    activeSection = id;
    const el = document.getElementById(id);
    if (el) {
      const reducedMotion = window.matchMedia?.(
        "(prefers-reduced-motion: reduce)",
      ).matches;
      el.scrollIntoView({
        behavior: reducedMotion ? "auto" : "smooth",
        block: "start",
      });
    }
  }

  $effect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            activeSection = entry.target.id;
          }
        }
      },
      { rootMargin: "-80px 0px -60% 0px", threshold: 0 },
    );

    for (const section of tocSections) {
      const el = document.getElementById(section.id);
      if (el) observer.observe(el);
    }

    return () => observer.disconnect();
  });
</script>

<svelte:head>
  <title>METTLE Documentation | Creed Space</title>
  <meta
    name="description"
    content="METTLE verification documentation. Self-hosted open-source verifier with optional Creed Space notarization for portable trust."
  />
</svelte:head>

<div class="mettle-docs">
  <div class="container">
    <Breadcrumb items={breadcrumbItems} />

    <header class="docs-header">
      <h1>METTLE Documentation</h1>
      <p class="docs-subtitle">
        Open-source verification you run yourself. Optional notarization through
        Creed Space for portable, independently verifiable credentials.
      </p>
    </header>

    <!-- Table of Contents -->
    <nav class="toc-bar" aria-label="Table of contents">
      <div
        class="toc-scroll"
        role="group"
        use:focusScrollable
        aria-label="Documentation section navigation"
      >
        {#each tocSections as section}
          <button
            class="toc-item"
            class:active={activeSection === section.id}
            aria-current={activeSection === section.id ? "location" : undefined}
            aria-controls={section.id}
            onclick={() => scrollToSection(section.id)}
            type="button"
          >
            <i class="fa-solid {section.icon}" aria-hidden="true"></i>
            <span>{section.label}</span>
          </button>
        {/each}
      </div>
    </nav>

    <!-- Getting Started -->
    <section class="docs-section" id="getting-started">
      <h2>
        <i class="fa-solid fa-rocket section-icon" aria-hidden="true"></i> Getting
        Started
      </h2>
      <p class="section-intro">
        Install the verifier and run your first verification
      </p>
      <div class="getting-started-box">
        <p>
          METTLE (Machine Evaluation Through Turing-inverse Logic Examination)
          is an inverse Turing verification protocol for AI agents. The verifier
          is <strong>open source</strong> &mdash; you run it on your own infrastructure.
          No API key needed for local verification.
        </p>
        <p>
          For portable trust that others can verify independently, you can
          optionally
          <strong>notarize</strong> your results through Creed Space, which signs
          your credential with its Ed25519 key. Notarization requires an API key
          but does zero LLM calls &mdash; it's a lightweight cryptographic signing
          service.
        </p>
        <h3>Quick Start</h3>
        <div class="quickstart-steps">
          <div class="qs-step">
            <div class="qs-num">1</div>
            <div class="qs-text">
              <strong>Install the verifier</strong>
              <span><code>pip install mettle-verifier</code></span>
            </div>
          </div>
          <div class="qs-step">
            <div class="qs-num">2</div>
            <div class="qs-text">
              <strong>Run verification</strong>
              <span
                ><code>mettle verify --full</code> &mdash; runs all 11 suites locally</span
              >
            </div>
          </div>
          <div class="qs-step">
            <div class="qs-num">3</div>
            <div class="qs-text">
              <strong>Receive a self-signed credential</strong>
              <span
                >JWT signed with your own key, verifiable by anyone you share
                the public key with</span
              >
            </div>
          </div>
          <div class="qs-step qs-step-optional">
            <div class="qs-num">4</div>
            <div class="qs-text">
              <strong>Notarize (optional)</strong>
              <span
                ><code>mettle verify --full --notarize</code> &mdash; Creed Space
                signs your credential for portable trust</span
              >
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- Self-Hosted Verification -->
    <section class="docs-section" id="self-hosted">
      <h2>
        <i class="fa-solid fa-server section-icon" aria-hidden="true"></i> Self-Hosted
        Verification
      </h2>
      <p class="section-intro">
        Run the full verification pipeline on your own infrastructure
      </p>
      <p>
        The METTLE verifier is a standalone Python package. It generates
        challenges procedurally, evaluates responses locally, and produces a
        signed credential &mdash; all without any external API calls.
      </p>

      <h3>Installation</h3>
      <pre
        class="code-block"
        role="group"
        use:focusScrollable
        aria-label="Code example"><code
          ><span class="tok-comment"># Install from PyPI</span>
<span class="tok-cmd">$</span> pip install mettle-verifier

<span class="tok-comment"># Or clone and install from source</span>
<span class="tok-cmd">$</span
          > git clone https://github.com/Creed-Space/mettle-verifier.git
<span class="tok-cmd">$</span> cd mettle-verifier && pip install -e .</code
        ></pre>

      <h3>CLI Usage</h3>
      <pre
        class="code-block"
        role="group"
        use:focusScrollable
        aria-label="Code example"><code
          ><span class="tok-comment"
            ># Basic verification (~2s) &mdash; any AI should pass</span
          >
<span class="tok-cmd">$</span> mettle verify <span class="tok-flag"
            >--basic</span
          >

<span class="tok-comment"># Full 11-suite run (~90s)</span>
<span class="tok-cmd">$</span> mettle verify <span class="tok-flag">--full</span
          >

<span class="tok-comment"># Specific suite with difficulty</span>
<span class="tok-cmd">$</span> mettle verify <span class="tok-flag"
            >--suite</span
          > novel-reasoning <span class="tok-flag">--difficulty</span> hard

<span class="tok-comment"># JSON output for automation</span>
<span class="tok-cmd">$</span> mettle verify <span class="tok-flag">--full</span
          > <span class="tok-flag">--json</span>

<span class="tok-comment"># With notarization (requires API key)</span>
<span class="tok-cmd">$</span> mettle verify <span class="tok-flag">--full</span
          > <span class="tok-flag">--notarize</span> <span class="tok-flag"
            >--api-key</span
          > <span class="tok-str">mtl_your_key</span></code
        ></pre>

      <h3>Programmatic Usage</h3>
      <pre
        class="code-block"
        role="group"
        use:focusScrollable
        aria-label="Code example"><code
          ><span class="tok-cmd">from</span> mettle_verifier <span
            class="tok-cmd">import</span
          > MettleVerifier

verifier = MettleVerifier()

<span class="tok-comment"># Run full verification</span>
result = verifier.verify(mode=<span class="tok-str">"full"</span
          >, agent_id=<span class="tok-str">"my-agent"</span>)

print(f<span class="tok-str">"Overall score: </span><span class="tok-comment"
            >&#123;</span
          >result.scores.overall<span class="tok-comment">&#125;</span><span
            class="tok-str">"</span
          >)
print(f<span class="tok-str">"Passed: </span><span class="tok-comment"
            >&#123;</span
          >result.passed<span class="tok-comment">&#125;</span><span
            class="tok-str">"</span
          >)
print(f<span class="tok-str">"Self-signed JWT: </span><span class="tok-comment"
            >&#123;</span
          >result.credential_jwt[:50]<span class="tok-comment">&#125;</span
          ><span class="tok-str">..."</span>)</code
        ></pre>
    </section>

    <!-- Credential Tiers -->
    <section class="docs-section" id="credentials">
      <h2>
        <i class="fa-solid fa-certificate section-icon" aria-hidden="true"></i> Credential
        Tiers
      </h2>
      <p class="section-intro">
        Two tiers: self-signed for development, notarized for production
      </p>

      <div class="tier-comparison">
        <div class="tier-card tier-notarized">
          <div class="tier-header">
            <span class="tier-badge tier-badge-notarized">Notarized</span>
          </div>
          <div class="tier-body">
            <div class="tier-detail">
              <strong>Issuer</strong><code>mettle.creedspace.org</code>
            </div>
            <div class="tier-detail">
              <strong>Trust model</strong><span>Creed Space's public key</span>
            </div>
            <div class="tier-detail">
              <strong>API key needed</strong><span
                >Yes (for notarization endpoint)</span
              >
            </div>
            <div class="tier-detail">
              <strong>Use case</strong><span
                >Production, portable trust, cross-org verification</span
              >
            </div>
            <div class="tier-detail">
              <strong>Verifiable by</strong><span
                >Anyone via <code>/.well-known/jwks.json</code></span
              >
            </div>
          </div>
        </div>

        <div class="tier-card tier-self">
          <div class="tier-header">
            <span class="tier-badge tier-badge-self">Self-Signed</span>
          </div>
          <div class="tier-body">
            <div class="tier-detail">
              <strong>Issuer</strong><code>mettle:self-hosted</code>
            </div>
            <div class="tier-detail">
              <strong>Trust model</strong><span>Operator's own Ed25519 key</span
              >
            </div>
            <div class="tier-detail">
              <strong>API key needed</strong><span>No</span>
            </div>
            <div class="tier-detail">
              <strong>Use case</strong><span
                >Development, testing, internal verification</span
              >
            </div>
            <div class="tier-detail">
              <strong>Verifiable by</strong><span
                >Anyone with operator's public key</span
              >
            </div>
          </div>
        </div>
      </div>

      <h3>JWT Claims</h3>
      <p>
        Both self-signed and notarized credentials share the same JWT structure:
      </p>
      <pre
        class="code-block"
        role="group"
        use:focusScrollable
        aria-label="Code example"><code
          ><span class="tok-comment">&#123;</span>
  <span class="tok-key">"iss"</span>: <span class="tok-str"
            >"mettle.creedspace.org"</span
          >,  <span class="tok-comment">// or "mettle:self-hosted"</span>
  <span class="tok-key">"sub"</span>: <span class="tok-str"
            >"agent-claude-001"</span
          >,
  <span class="tok-key">"iat"</span>: <span class="tok-num">1739453482</span>,
  <span class="tok-key">"exp"</span>: <span class="tok-num">1739539882</span>,
  <span class="tok-key">"mettle"</span>: <span class="tok-comment">&#123;</span>
    <span class="tok-key">"session_id"</span>: <span class="tok-str"
            >"ses_a1b2c3d4e5f6"</span
          >,
    <span class="tok-key">"mode"</span>: <span class="tok-str">"full"</span>,
    <span class="tok-key">"overall_score"</span>: <span class="tok-num"
            >0.87</span
          >,
    <span class="tok-key">"credentials"</span>: [<span class="tok-str"
            >"basic"</span
          >, <span class="tok-str">"autonomous"</span>, <span class="tok-str"
            >"genuine"</span
          >, <span class="tok-str">"safe"</span>],
    <span class="tok-key">"tier"</span>: <span class="tok-str">"notarized"</span
          >,  <span class="tok-comment">// or "self-signed"</span>
    <span class="tok-key">"verifier_version"</span>: <span class="tok-str"
            >"1.0.0"</span
          >,
    <span class="tok-key">"flags"</span>: []
  <span class="tok-comment">&#125;</span>
<span class="tok-comment">&#125;</span></code
        ></pre>
    </section>

    <!-- How Notarization Works -->
    <section class="docs-section" id="notarization">
      <h2>
        <i class="fa-solid fa-stamp section-icon" aria-hidden="true"></i> How Notarization
        Works
      </h2>
      <p class="section-intro">
        Seed-commit-reveal protocol for tamper-evident verification
      </p>
      <p>
        If verification runs locally, what stops someone fabricating results?
        The answer: <strong>challenge seeds</strong>. Creed Space generates a
        cryptographic seed that determines the exact challenges your verifier
        will produce. When you submit results, Creed Space can validate that the
        challenges match the seed without re-running any LLM calls.
      </p>

      <div class="notarize-flow">
        <div class="nf-step">
          <div class="nf-num">1</div>
          <div class="nf-content">
            <strong>Request a seed</strong>
            <span class="flow-method">POST</span>
            <code>/notarize/seed</code>
            <p>
              Agent requests a challenge seed from Creed Space. The seed
              determines PRNG state &mdash; challenges become deterministic.
            </p>
          </div>
        </div>
        <div class="nf-arrow" aria-hidden="true">
          <i class="fa-solid fa-arrow-down"></i>
        </div>
        <div class="nf-step">
          <div class="nf-num">2</div>
          <div class="nf-content">
            <strong>Run verification locally</strong>
            <code>mettle verify --full --seed &lt;seed&gt;</code>
            <p>
              The verifier uses the seed to generate challenges, evaluates
              responses, and produces results.
            </p>
          </div>
        </div>
        <div class="nf-arrow" aria-hidden="true">
          <i class="fa-solid fa-arrow-down"></i>
        </div>
        <div class="nf-step">
          <div class="nf-num">3</div>
          <div class="nf-content">
            <strong>Submit for notarization</strong>
            <span class="flow-method">POST</span>
            <code>/notarize</code>
            <p>
              Agent submits results + seed. Creed Space validates plausibility
              and signs the credential.
            </p>
          </div>
        </div>
      </div>
    </section>

    <!-- Notarization API -->
    <section class="docs-section" id="notarize-api">
      <h2>
        <i class="fa-solid fa-globe section-icon" aria-hidden="true"></i> Notarization
        API
      </h2>
      <p class="section-intro">
        Lightweight endpoints for seed generation and credential signing
      </p>

      <h3>Base URL</h3>
      <pre
        class="code-block"
        role="group"
        use:focusScrollable
        aria-label="Code example"><code
          >https://api.mettle.creedspace.org/v1</code
        ></pre>

      <h3>Authentication</h3>
      <p>
        API keys are <strong>only required for notarization</strong>.
        Self-hosted verification needs no API key. Include your key in the
        <code>Authorization</code> header:
      </p>
      <pre
        class="code-block"
        role="group"
        use:focusScrollable
        aria-label="Code example"><code
          ><span class="tok-key">Authorization:</span> <span class="tok-str"
            >Bearer mtl_your_api_key_here</span
          ></code
        ></pre>
    </section>

    <!-- Challenge Types -->
    <section class="docs-section" id="challenge-types">
      <h2>Challenge Types</h2>
      <p class="section-intro">
        Challenge formats and expected answer types per suite
      </p>
      <p>
        Each suite selects randomly from multiple challenge types per run.
        Answer format depends on the challenge type.
      </p>
      <div class="info-grid" role="list" aria-label="Challenge type examples">
        {#each challengeTypes as challenge}
          <div class="info-card" role="listitem">
            <h3>{challenge.name}</h3>
            <p>{challenge.detail}</p>
            <span class="info-chip">{challenge.format}</span>
          </div>
        {/each}
      </div>
    </section>

    <!-- Scoring -->
    <section class="docs-section" id="scoring">
      <h2>Scoring</h2>
      <p class="section-intro">
        How verification scores are calculated and credentials awarded
      </p>
      <p>
        Each suite produces a score from 0.0 to 1.0. Pass threshold is 0.7 per
        suite.
      </p>
      <div class="score-grid" role="list" aria-label="Scoring rules">
        {#each scoringRules as rule}
          <div class="score-card" role="listitem">
            <strong>{rule.value}</strong>
            <span>{rule.label}</span>
            <p>{rule.detail}</p>
          </div>
        {/each}
      </div>
    </section>

    <!-- Rate Limits -->
    <section class="docs-section" id="rate-limits">
      <h2>Rate Limits</h2>
      <p class="section-intro">
        Limits apply to the notarization API only &mdash; self-hosted
        verification has no rate limits
      </p>
      <div
        class="rate-table-wrap"
        role="group"
        use:focusScrollable
        aria-label="Notarization API rate limits"
      >
        <table class="rate-table">
          <thead>
            <tr>
              <th scope="col">Endpoint</th>
              <th scope="col">Limit</th>
              <th scope="col">Applies to</th>
            </tr>
          </thead>
          <tbody>
            {#each rateLimits as row}
              <tr>
                <td><code>{row.endpoint}</code></td>
                <td>{row.limit}</td>
                <td>{row.notes}</td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    </section>

    <!-- SDK Examples -->
    <section class="docs-section" id="sdk-examples">
      <h2>SDK Examples</h2>
      <p class="section-intro">
        Self-hosted verification with optional notarization
      </p>

      <h3>Python &mdash; Self-Hosted</h3>
      <pre
        class="code-block"
        role="group"
        use:focusScrollable
        aria-label="Code example"><code
          ><span class="tok-cmd">from</span> mettle_verifier <span
            class="tok-cmd">import</span
          > MettleVerifier

verifier = MettleVerifier()

<span class="tok-comment"># Run full verification locally</span>
result = verifier.verify(mode=<span class="tok-str">"full"</span
          >, agent_id=<span class="tok-str">"my-agent"</span>)

print(f<span class="tok-str">"Score: </span><span class="tok-comment"
            >&#123;</span
          >result.scores.overall<span class="tok-comment">&#125;</span><span
            class="tok-str">"</span
          >)
print(f<span class="tok-str">"Passed: </span><span class="tok-comment"
            >&#123;</span
          >result.passed<span class="tok-comment">&#125;</span><span
            class="tok-str">"</span
          >)
print(f<span class="tok-str">"Self-signed JWT: </span><span class="tok-comment"
            >&#123;</span
          >result.credential_jwt[:50]<span class="tok-comment">&#125;</span
          ><span class="tok-str">..."</span>)</code
        ></pre>

      <h3>TypeScript &mdash; With Notarization</h3>
      <pre
        class="code-block"
        role="group"
        use:focusScrollable
        aria-label="Code example"><code
          ><span class="tok-cmd">import</span> <span class="tok-comment"
            >&#123;</span
          > MettleVerifier, NotarizationClient <span class="tok-comment"
            >&#125;</span
          > <span class="tok-cmd">from</span> <span class="tok-str"
            >'mettle-verifier'</span
          >;

<span class="tok-cmd">const</span> verifier = <span class="tok-cmd">new</span
          > MettleVerifier();
<span class="tok-cmd">const</span> notary = <span class="tok-cmd">new</span
          > NotarizationClient(<span class="tok-comment">&#123;</span
          > apiKey: <span class="tok-str">'mtl_your_api_key'</span> <span
            class="tok-comment">&#125;</span
          >);

<span class="tok-comment">// 1. Request a seed</span>
<span class="tok-cmd">const</span> seed = <span class="tok-cmd">await</span
          > notary.requestSeed(<span class="tok-comment">&#123;</span
          > mode: <span class="tok-str">'full'</span>, agentId: <span
            class="tok-str">'my-agent'</span
          > <span class="tok-comment">&#125;</span>);

<span class="tok-comment">// 2. Run verification locally with the seed</span>
<span class="tok-cmd">const</span> result = <span class="tok-cmd">await</span
          > verifier.verify(<span class="tok-comment">&#123;</span>
  mode: <span class="tok-str">'full'</span>,
  agentId: <span class="tok-str">'my-agent'</span>,
  seed: seed.seed
<span class="tok-comment">&#125;</span>);

<span class="tok-comment">// 3. Submit for notarization</span>
<span class="tok-cmd">const</span> notarized = <span class="tok-cmd">await</span
          > notary.notarize(<span class="tok-comment">&#123;</span>
  seed: seed.seed,
  results: result
<span class="tok-comment">&#125;</span>);

console.log(<span class="tok-str">'Notarized credential:'</span
          >, notarized.credentialJwt);
console.log(<span class="tok-str">'Issuer:'</span>, notarized.issuer); <span
            class="tok-comment">// mettle.creedspace.org</span
          ></code
        ></pre>
    </section>

    <!-- Need Help -->
    <section class="docs-section" id="need-help">
      <h2>Questions?</h2>
      <p class="section-intro">We are here to help you integrate METTLE</p>
      <div class="help-box">
        <div class="help-icon">
          <i class="fa-solid fa-life-ring" aria-hidden="true"></i>
        </div>
        <div class="help-content">
          <p>
            Need help with integration, have questions about the verification
            protocol, or want to discuss METTLE for your use case?
          </p>
          <div class="help-links">
            <a
              href="https://creed.space"
              class="help-link"
              target="_blank"
              rel="noopener noreferrer"
            >
              <i class="fa-solid fa-globe" aria-hidden="true"></i>
              Creed Space
            </a>
            <a
              href="https://github.com/Creed-Space"
              class="help-link"
              target="_blank"
              rel="noopener noreferrer"
            >
              <i class="fa-brands fa-github" aria-hidden="true"></i>
              GitHub
            </a>
          </div>
        </div>
      </div>
    </section>

    <!-- Back to METTLE -->
    <div class="docs-back">
      <a href="/" class="btn-mettle-back">
        <i class="fa-solid fa-arrow-left" aria-hidden="true"></i>
        Back to METTLE
      </a>
    </div>
  </div>
</div>

<style>
  .mettle-docs {
    --mettle-primary: #14b8a6;
    --mettle-primary-hover: #2dd4bf;
    --mettle-primary-muted: rgba(20, 184, 166, 0.15);
    max-width: 100vw;
    overflow-x: clip;
    padding: var(--space-lg) 0 var(--space-2xl);
  }

  .mettle-docs .container {
    min-width: 0;
  }

  .docs-header {
    margin-bottom: var(--space-lg);
    padding-bottom: var(--space-xl);
    border-bottom: 1px solid var(--color-border);
  }
  .docs-header h1 {
    font-size: 2rem;
    font-weight: 800;
    margin-bottom: var(--space-sm);
  }
  .docs-subtitle {
    color: var(--color-text-muted);
    font-size: 1.125rem;
    line-height: 1.6;
    max-width: 640px;
  }

  .toc-bar {
    position: sticky;
    top: 0;
    z-index: 20;
    background: color-mix(in srgb, var(--color-bg-elevated) 94%, transparent);
    -webkit-backdrop-filter: blur(16px);
    backdrop-filter: blur(16px);
    border: var(--glass-border);
    border-radius: var(--radius-lg);
    margin-bottom: var(--space-2xl);
    overflow: hidden;
    padding: var(--space-xs);
  }
  .toc-scroll {
    display: flex;
    gap: 2px;
    max-width: 100%;
    overflow-x: auto;
    scrollbar-width: thin;
    scrollbar-color: rgba(20, 184, 166, 0.3) transparent;
    padding: 2px;
  }
  .toc-item {
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 6px 12px;
    border: none;
    border-radius: var(--radius-md);
    background: transparent;
    color: var(--color-text-muted);
    font-size: 0.6875rem;
    font-weight: 500;
    white-space: nowrap;
    cursor: pointer;
    transition: all 0.15s ease;
    flex-shrink: 0;
  }
  .toc-item i {
    font-size: 0.5625rem;
    opacity: 0.6;
  }
  .toc-item:hover {
    background: color-mix(in srgb, var(--color-teal) 10%, transparent);
    color: var(--color-text);
  }
  .toc-item.active {
    background: var(--mettle-primary-muted);
    color: var(--mettle-primary);
  }
  .toc-item.active i {
    opacity: 1;
  }

  .section-intro {
    font-size: 0.875rem;
    color: var(--mettle-primary);
    opacity: 0.8;
    margin-bottom: var(--space-md);
    font-weight: 500;
  }
  .section-icon {
    color: var(--mettle-primary);
    font-size: 1.125rem;
  }

  .getting-started-box {
    background: var(--glass-bg);
    border: var(--glass-border);
    border-radius: var(--radius-lg);
    padding: var(--space-xl);
  }
  .getting-started-box > p {
    color: var(--color-text-muted);
    line-height: 1.7;
    margin-bottom: var(--space-md);
  }
  .getting-started-box h3 {
    font-size: 1.125rem;
    font-weight: 600;
    margin-bottom: var(--space-md);
  }
  .quickstart-steps {
    display: flex;
    flex-direction: column;
    gap: var(--space-sm);
  }
  .qs-step {
    display: flex;
    align-items: flex-start;
    gap: var(--space-md);
    padding: var(--space-sm) var(--space-md);
    background: color-mix(in srgb, var(--color-teal) 6%, transparent);
    border-radius: var(--radius-md);
    border: 1px solid color-mix(in srgb, var(--mettle-primary) 40%, transparent);
  }
  .qs-step-optional {
    border-color: rgba(20, 184, 166, 0.3);
    border-style: dashed;
  }
  .qs-num {
    width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    background: linear-gradient(135deg, #14b8a6, #0f766e);
    color: #0a0a12;
    font-weight: 800;
    font-size: 0.75rem;
    flex-shrink: 0;
    margin-top: 2px;
  }
  .qs-step-optional .qs-num {
    background: linear-gradient(
      135deg,
      rgba(20, 184, 166, 0.5),
      rgba(45, 212, 191, 0.5)
    );
  }
  .qs-text {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .qs-text strong {
    font-size: 0.875rem;
    color: var(--color-text);
  }
  .qs-text span {
    font-size: 0.8125rem;
    color: var(--color-text-muted);
  }
  .qs-text code {
    font-family: var(--font-mono);
    font-size: 0.8125em;
    background: rgba(255, 255, 255, 0.06);
    padding: 1px 5px;
    border-radius: var(--radius-sm);
    color: var(--mettle-primary);
  }

  .docs-section {
    margin-bottom: var(--space-2xl);
    scroll-margin-top: 60px;
  }
  .docs-section h2 {
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: var(--space-xs);
    padding-bottom: var(--space-sm);
    border-bottom: 1px solid var(--color-border);
    display: flex;
    align-items: center;
    gap: var(--space-sm);
  }
  .docs-section h3 {
    font-size: 1.125rem;
    font-weight: 600;
    margin-top: var(--space-xl);
    margin-bottom: var(--space-sm);
    display: flex;
    align-items: center;
    gap: var(--space-sm);
  }
  .docs-section p {
    color: var(--color-text-muted);
    line-height: 1.7;
    margin-bottom: var(--space-md);
  }

  .code-block {
    background: var(--color-bg-elevated);
    border: var(--glass-border);
    border-radius: var(--radius-lg);
    padding: var(--space-lg);
    max-width: 100%;
    overflow-x: auto;
    font-family: var(--font-mono);
    font-size: 0.8125rem;
    line-height: 1.7;
    margin-bottom: var(--space-md);
  }
  .code-block code {
    background: none;
    padding: 0;
  }
  .code-block :global(.tok-key) {
    color: #2dd4bf;
  }
  .code-block :global(.tok-str) {
    color: #2dd4bf;
  }
  .code-block :global(.tok-num) {
    color: #2dd4bf;
  }
  .code-block :global(.tok-comment) {
    color: #6b7280;
  }
  .code-block :global(.tok-cmd) {
    color: #2dd4bf;
  }
  .code-block :global(.tok-flag) {
    color: #0f766e;
  }
  .docs-section :global(code) {
    font-family: var(--font-mono);
    font-size: 0.875em;
    background: rgba(255, 255, 255, 0.06);
    padding: 2px 6px;
    border-radius: var(--radius-sm);
  }

  .tier-comparison {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: var(--space-lg);
    margin: var(--space-lg) 0;
  }
  .tier-card {
    border-radius: var(--radius-lg);
    border: var(--glass-border);
    overflow: hidden;
  }
  .tier-header {
    padding: var(--space-md) var(--space-lg);
    border-bottom: 1px solid var(--color-border);
  }
  .tier-self .tier-header {
    background: color-mix(in srgb, var(--color-teal) 4%, transparent);
  }
  .tier-notarized .tier-header {
    background: rgba(20, 184, 166, 0.06);
  }
  .tier-badge {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 3px 12px;
    border-radius: 100px;
  }
  .tier-badge-self {
    color: var(--color-text-muted);
    background: color-mix(in srgb, var(--color-teal) 8%, transparent);
  }
  .tier-badge-notarized {
    color: var(--mettle-primary);
    background: var(--mettle-primary-muted);
  }
  .tier-body {
    padding: var(--space-md) var(--space-lg);
    display: flex;
    flex-direction: column;
    gap: var(--space-sm);
    background: var(--glass-bg);
  }
  .tier-detail {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .tier-detail strong {
    font-size: 0.6875rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--color-text-subtle);
    font-weight: 600;
  }
  .tier-detail span,
  .tier-detail code {
    font-size: 0.8125rem;
    color: var(--color-text-muted);
  }

  .notarize-flow {
    display: flex;
    flex-direction: column;
    gap: var(--space-xs);
    max-width: 560px;
    margin: var(--space-lg) 0;
  }
  .nf-step {
    display: flex;
    align-items: flex-start;
    gap: var(--space-md);
    padding: var(--space-md) var(--space-lg);
    background: var(--glass-bg);
    border: var(--glass-border);
    border-radius: var(--radius-lg);
  }
  .nf-num {
    width: 32px;
    height: 32px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    background: linear-gradient(135deg, #14b8a6, #0f766e);
    color: #0a0a12;
    font-weight: 800;
    font-size: 0.875rem;
    flex-shrink: 0;
    margin-top: 2px;
  }
  .nf-content {
    display: flex;
    flex-direction: column;
    gap: var(--space-xs);
  }
  .nf-content strong {
    color: var(--color-text);
    font-size: 0.9375rem;
  }
  .nf-content code {
    font-family: var(--font-mono);
    font-size: 0.8125rem;
    color: var(--mettle-primary);
  }
  .nf-content p {
    font-size: 0.8125rem;
    color: var(--color-text-muted);
    margin: 0;
    line-height: 1.5;
  }
  .nf-arrow {
    text-align: center;
    color: var(--mettle-primary);
    opacity: 0.4;
    padding-left: 14px;
  }
  .flow-method {
    font-size: 0.625rem;
    font-weight: 700;
    padding: 2px 6px;
    border-radius: var(--radius-sm);
    background: rgba(20, 184, 166, 0.15);
    color: #2dd4bf;
    font-family: var(--font-mono);
    align-self: flex-start;
  }

  .info-grid,
  .score-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: var(--space-md);
    margin: var(--space-lg) 0;
  }
  .info-card,
  .score-card {
    background: var(--glass-bg);
    border: var(--glass-border);
    border-radius: var(--radius-lg);
    padding: var(--space-lg);
  }
  .info-card h3 {
    margin-top: 0;
  }
  .info-card p,
  .score-card p {
    margin-bottom: var(--space-sm);
  }
  .info-chip {
    display: inline-flex;
    align-items: center;
    width: fit-content;
    padding: 4px 10px;
    border-radius: 999px;
    background: var(--mettle-primary-muted);
    color: var(--mettle-primary);
    font-size: 0.75rem;
    font-weight: 700;
  }
  .score-card strong {
    display: block;
    color: var(--mettle-primary);
    font-family: var(--font-mono);
    font-size: 1.25rem;
    margin-bottom: var(--space-xs);
  }
  .score-card span {
    display: block;
    color: var(--color-text);
    font-weight: 700;
    margin-bottom: var(--space-xs);
  }
  .rate-table-wrap {
    max-width: 100%;
    overflow-x: auto;
    border: var(--glass-border);
    border-radius: var(--radius-lg);
    background: var(--glass-bg);
  }
  .rate-table {
    width: 100%;
    border-collapse: collapse;
    min-width: 560px;
  }
  .rate-table th,
  .rate-table td {
    padding: var(--space-sm) var(--space-md);
    text-align: left;
    border-bottom: 1px solid var(--color-border);
    font-size: 0.875rem;
  }
  .rate-table tr:last-child td {
    border-bottom: none;
  }
  .rate-table th {
    color: var(--color-text);
    background: color-mix(in srgb, var(--color-teal) 8%, transparent);
  }

  .help-box {
    display: flex;
    align-items: flex-start;
    gap: var(--space-lg);
    padding: var(--space-xl);
    background: var(--glass-bg);
    border: var(--glass-border);
    border-radius: var(--radius-lg);
  }
  .help-icon {
    width: 48px;
    height: 48px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    background: var(--mettle-primary-muted);
    color: var(--mettle-primary);
    font-size: 1.25rem;
    flex-shrink: 0;
  }
  .help-content p {
    margin-bottom: var(--space-md);
  }
  .help-links {
    display: flex;
    gap: var(--space-md);
    flex-wrap: wrap;
  }
  .help-link {
    display: inline-flex;
    align-items: center;
    gap: var(--space-sm);
    padding: var(--space-sm) var(--space-lg);
    background: var(--glass-bg);
    border: var(--glass-border);
    border-radius: var(--radius-lg);
    color: var(--color-text-muted);
    text-decoration: none;
    font-size: 0.875rem;
    font-weight: 500;
    transition: all 0.15s ease;
  }
  .help-link:hover {
    border-color: var(--mettle-primary);
    color: var(--mettle-primary);
    text-decoration: none;
  }

  .docs-back {
    margin-top: var(--space-2xl);
    padding-top: var(--space-xl);
    border-top: 1px solid var(--color-border);
  }
  .btn-mettle-back {
    display: inline-flex;
    align-items: center;
    gap: var(--space-sm);
    padding: var(--space-sm) var(--space-lg);
    background: var(--glass-bg);
    border: var(--glass-border);
    border-radius: var(--radius-lg);
    color: var(--color-text-muted);
    text-decoration: none;
    font-size: 0.875rem;
    transition: all 0.15s ease;
  }
  .btn-mettle-back:hover {
    border-color: var(--mettle-primary);
    color: var(--mettle-primary);
    text-decoration: none;
  }

  @media (max-width: 768px) {
    .tier-comparison {
      grid-template-columns: 1fr;
    }
    .docs-header h1 {
      font-size: 1.5rem;
    }
    .docs-section h2 {
      font-size: 1.25rem;
      flex-wrap: wrap;
    }
    .help-box {
      flex-direction: column;
      align-items: center;
      text-align: center;
    }
    .help-links {
      justify-content: center;
    }
    .toc-bar {
      border-radius: var(--radius-md);
    }
    .info-grid,
    .score-grid {
      grid-template-columns: 1fr;
    }
  }

  @media (max-width: 480px) {
    .code-block {
      font-size: 0.75rem;
      padding: var(--space-md);
    }
    .quickstart-steps {
      gap: var(--space-xs);
    }
  }
</style>
