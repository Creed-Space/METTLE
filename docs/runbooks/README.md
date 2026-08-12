# METTLE operations runbooks

These runbooks define safe first actions, owner roles, evidence, and recovery
gates. They do not grant production authority. Run provider mutations only from
an authenticated operator session with explicit incident ownership.

| Incident | Runbook | Primary owner role |
|---|---|---|
| Redis unavailable, divergent, or recovered | [Redis loss](REDIS_LOSS.md) | Runtime operator |
| PostgreSQL unavailable, corrupt, or restored | [Database loss](DATABASE_LOSS.md) | Data operator |
| Ed25519 or JWT signing material suspected compromised | [Signing-key compromise](SIGNING_KEY_COMPROMISE.md) | Security incident commander |
| Abusive traffic or credential probing | [Abusive traffic](ABUSIVE_TRAFFIC.md) | Security operations |
| False accept or false reject spike | [False-decision spike](FALSE_DECISION_SPIKE.md) | Protocol maintainer |
| Bad deployment or rollback | [Deployment rollback](DEPLOYMENT_ROLLBACK.md) | Release operator |
| New or rotated public signing key | [Public-key publication](PUBLIC_KEY_PUBLICATION.md) | Credential operator |
| Backup restoration or unrecoverable key loss | [Backup restore and key loss](BACKUP_RESTORE_AND_KEY_LOSS.md) | Data and credential operators |

## Common receipt

Every incident retains incident ID, UTC start and end, owner, exact source and
deployed SHA, protocol versions, topology, provider event IDs, redacted request
IDs, observed impact, commands or dashboard actions, before and after health,
decision log, data impact, key impact, rollback target, and follow-up disposition.

Never retain secrets, raw challenge answers, credentials, authorization headers,
full database URLs, or unnecessary participant identity in an incident receipt.

Working if: the on-call operator can choose the correct runbook from a symptom,
every provider mutation has an identified owner, and closure evidence proves the
same deployed candidate rather than a local approximation.
