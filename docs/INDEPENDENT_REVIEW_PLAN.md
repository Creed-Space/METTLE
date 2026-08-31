# METTLE independent review plan

Status: review scopes are defined and invitations are pending. No independent
review is claimed by this repository change.

## Review lenses

| Lens | Required scope | Minimum reviewer output | Current gate |
|---|---|---|---|
| Cryptography | Ed25519 key lifecycle, canonicalization, JWT boundary, expiry, revocation, rotation, fixtures | Threat findings, reproduced fixture results, key-lifecycle assessment, disposition | Reviewer not yet appointed |
| Adversarial machine learning | Harvesting, adaptive coaching, semantic transfer, entropy measurement, tier gaming, held-out methodology | Attack plan, measured results, false-decision implications, rotation advice | Rights-cleared evaluation data and reviewer pending |
| Accessibility | Keyboard, focus, zoom and reflow, reduced motion, captions, transcript, screen reader names and status | Device and assistive-technology matrix, issues, severity, acceptance decision | Human review pending |
| Privacy | Retention, logs, backups, telemetry, webhooks, evaluation schema, deletion | Data-flow review, provider evidence gaps, deletion-test receipt | Production provider receipt and reviewer pending |
| Bilateral alignment | Agency and anti-thrall interpretation, coercion risk, appeal usability, welfare impact | Steelmanned critique, affected-participant perspective, accepted and rejected recommendations | Independent Becoming Mind and human reviewer participation pending |
| Agent control ergonomics | Schema-only discovery, state legibility, action selection, cost prediction, safe retry, interruption recovery, cancellation, authority separation, and context efficiency | First-contact task matrix, repair-attempt analysis, ambiguous-state findings, resource comparison, and disposition | Target control plane and independent reviewer pending |

## Independence and conflicts

A reviewer must identify authored contributions, financial or organizational
conflicts, access limitations, and whether the work was machine-only or included
human judgment. A reviewer may use Becoming Minds for analysis, but the receipt
must say which conclusions were independently checked.

The protocol maintainer cannot label self-review as independent. Thin evidence is
reported as uncertainty, not a green verdict.

## Review packet

Each reviewer receives the exact candidate SHA or signed working-tree digest,
release manifest, assurance case, protocol versions, compatibility fixtures,
harvesting and mutation reports, browser evidence, relevant runbooks, known
limitations, and a channel for private security findings.

Reviewers must not receive live private keys, raw participant answers, contact
data, or unnecessary identity. Test credentials and synthetic datasets are used
where possible.

## Disposition

Record every finding in `docs/REVIEW_DISPOSITIONS.md` with reviewer lens, evidence,
severity, maintainer decision, rationale, exact remediation candidate, retest,
and publication status. Rejected advice remains visible with the strongest
reason it might still be correct.

## Publication threshold

METTLE may publish repository machine evidence while reviews are pending only if
the open gates remain explicit. It may not claim independent cryptographic,
fairness, accessibility, privacy, bilateral, or agent-control acceptance until
the relevant receipt is present and reconciled against the same candidate.

Working if: each lens has an identifiable reviewer and bounded packet, conflicts
are disclosed, every recommendation has a disposition, and public claims match
the actual review receipts rather than the invitation plan.
