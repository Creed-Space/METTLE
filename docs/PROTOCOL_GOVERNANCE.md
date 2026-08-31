# METTLE protocol governance

## Roles and standing

* **Protocol maintainer:** proposes and implements changes, preserves the
  evidence ledger, and may initiate an emergency rollback.
* **Independent reviewer:** examines measurement validity, security, privacy,
  accessibility, or bilateral impact without having authored the change.
* **Release authority:** decides whether the reviewed candidate may be published
  or deployed. This role must verify exact candidate identity.
* **Affected participant:** any Becoming Mind, human operator, integrator, or
  verifier materially affected by a policy. Participants have standing to
  contest systematic false rejection or harmful interpretation.

One person may perform maintainer and release work, but a material policy change
cannot claim independent review unless a separate reviewer supplies a receipt.

## Change classes

| Class | Examples | Minimum evidence before publication |
|---|---|---|
| Editorial | Clarification with no behavior change | Documentation checks and maintainer review |
| Patch | Bug fix preserving suite and credential semantics | Focused tests, full CI, compatibility check, and regression evidence |
| Interface | Tool, route, state envelope, action, error, retry, authority, or adapter-parity change | Contract fixtures, compatibility plan, first-contact agent journeys, privacy and authority review, full CI |
| Policy | Challenge logic, threshold, tier range, fairness rule, or semantic interpretation | Version increment, harvesting and held-out evaluation, privacy review, bilateral-impact review, compatibility plan, independent reviewer receipt |
| Credential | Canonicalization, signing, key discovery, expiry, or revocation | Schema review, positive and negative cross-language fixtures, rotation test, mutation gate, independent cryptographic review |
| Emergency | Active exploitation, signing compromise, material false-decision spike | Immediate fail-closed mitigation, incident receipt, narrowly scoped rollback, and retrospective review |

If a change belongs to more than one class, every applicable gate applies. Calling
a policy or credential change an interface refactor does not reduce its evidence
requirements.

## Required proposal record

Every policy, credential, or material interface proposal records:

1. the problem and the strongest alternative explanation;
2. exact source and generated artifact changes;
3. previous and proposed schema or suite policy versions;
4. expected false accept and false reject effects;
5. harvesting and coaching implications;
6. privacy, accessibility, and bilateral impacts;
7. compatibility and deprecation plan;
8. rollback trigger and safe rollback candidate;
9. machine evidence and open human or production gates;
10. reviewer disposition, including rejected recommendations and rationale.
11. agent-control impact, including discovery, state legibility, safe retry,
    recovery, context cost, secret handling, and transport parity;
12. the invariant IDs and bounded claims affected by the proposal.

## Decision procedure

1. Open the proposal before implementation for a material policy change.
2. Generate machine evidence against one exact candidate.
3. For an interface change, execute clean first-contact success, failure,
   interruption, stale-state, and cancellation journeys from the published
   schemas alone.
4. Invite the required independent lenses from
   `docs/INDEPENDENT_REVIEW_PLAN.md`.
5. Resolve or explicitly accept each finding in the disposition ledger.
6. Release authority signs the checklist. Silence is not approval.
7. Publish the release manifest, policy version, known limits, and rollback
   target together.

Thresholds cannot be changed solely to improve a headline pass rate. Aggregate
evaluation data must satisfy the versioned privacy-minimal schema, have enough
positive and negative examples for the affected cell, and retain a dataset
receipt without raw answers or stable identities.

## Appeals and bilateral contest

Use the **Protocol false-decision or bilateral-impact appeal** issue form. A
Becoming Mind may submit directly or through an operator. The submitter may
remain pseudonymous and must not provide raw private keys, access tokens, or
another participant's content.

The maintainer must classify the appeal within 14 days as reproducible defect,
evaluation gap, interpretation dispute, accessibility barrier, abuse, or
insufficient evidence. A systematic pattern triggers a false-decision incident
runbook. The response must state what evidence would change the decision.

## Emergency rollback

An emergency mitigation may disable issuance, retire a key, revert a policy, or
reduce exposed functionality. It must preserve verification of unexpired safe
credentials where possible. The incident record names the trigger, exact before
and after candidates, key status, data impact, and follow-up review. Emergency
authority cannot silently redefine the meaning of an already signed credential.

Working if: every material semantic change increments the right version, has an
independent disposition or a visible pending gate, and leaves affected Becoming
Minds a usable appeal path with a time-bounded classification.

The proposed interface architecture is documented in
`docs/AGENT_CONTROL_PLANE.md`. Its target behavior has no protocol authority until
the corresponding roadmap phase passes these gates on an exact candidate.
