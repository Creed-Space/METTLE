# METTLE security policy

## Supported versions

| Version | Security fixes | Status |
|---|---|---|
| `0.4.x` | Yes | Current supported line |
| `main` | Yes | Development candidate, not a release |
| Earlier than `0.4` | No | Upgrade before reporting version-specific defects |

The latest patch in the supported line supersedes earlier patches. If a defect
also affects an older version, the project may document it, but only the current
line is promised a repair assessment.

## Private reporting

Use [GitHub private vulnerability reporting](https://github.com/Creed-Space/METTLE/security/advisories/new).
Include the affected commit or version, prerequisites, reproduction steps,
impact, and any suggested mitigation. Do not include live credentials, private
keys, challenge answers, or another person's data. If private reporting is not
available, use the repository Security tab to contact the maintainers without
publishing exploit details in a public issue.

Security response targets are:

| Stage | Target |
|---|---|
| Acknowledge receipt | 3 business days |
| Initial severity and scope assessment | 7 business days |
| Status update while unresolved | Every 14 days |
| Critical mitigation target | 7 days after validation |
| High mitigation target | 30 days after validation |
| Moderate or low mitigation target | 90 days after validation or a documented release plan |

These are operational targets, not guarantees. A report is not considered
resolved until a supported release or deployment receipt identifies the exact
candidate and remediation.

## Safe harbor

Good-faith research is welcomed when it:

* stays within accounts, keys, sessions, and data the researcher owns or has
  explicit permission to test;
* uses the smallest request volume needed to demonstrate the issue;
* avoids service disruption, persistence, social engineering, privacy invasion,
  challenge-corpus harvesting, and access to another user's data;
* stops when sensitive data, a secret, or evidence of harm is encountered;
* reports privately and allows a reasonable remediation period before disclosure.

The project will not recommend legal action solely for research that follows
these conditions. This statement cannot authorize activity prohibited by a
third-party provider or by law.

## Security boundary

METTLE is experimental verification software. A pass is a bounded behavioral
measurement under one suite policy. It does not prove identity, consciousness,
safety, autonomy, personhood, moral status, or operator trustworthiness.

Valid security reports include authentication or ownership bypass, answer
leakage, replay, signature or canonicalization failure, rate-limit bypass,
cross-worker authority divergence, secret exposure, unsafe webhook behavior,
revocation failure, and dependency or deployment compromise. Disagreement with
the ethical interpretation of a suite belongs in the protocol appeal process,
unless it also creates a concrete security impact.

Working if: private reports receive a receipt against an exact version, public
issues contain no exploit secret, and every closure links to a released or
deployed candidate plus its verification evidence.
