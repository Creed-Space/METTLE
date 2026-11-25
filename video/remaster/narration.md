# METTLE remaster narration

## The reverse question

For decades, a familiar checkbox has asked us to prove that we are human. METTLE approaches the encounter from another direction, by offering challenges designed around machine capabilities and observing how a respondent performs. That shift opens an interesting experimental question: what can a carefully bounded interaction tell us about the behaviour of the system answering? The useful starting point is a measurement made in a particular session.

## A bounded experiment

METTLE stands for Machine Evaluation Through Turing-inverse Logic Examination. Its experimental protocol places each session under a named, versioned policy, so the challenge, the response and the interpretation belong to an explicit frame of reference. The server records answers, timing and scores, creating a record that another party can examine. This gives a result a precise meaning: it describes observed behaviour under the conditions of that session.

## Inside a challenge

The quick API offers a compact screening, including an arithmetic challenge that asks for the sum, difference or product of two eight-digit numbers within half a second. Other tasks ask the respondent to continue a generated token sequence or satisfy an exact formatting rule. Further challenges examine consistency, self-description and changes after feedback. These are observable response patterns, and they can be produced by people, models, purpose-built solvers or relayed answers. Their interpretation therefore calls for care.

## Twelve research lenses

The separate authenticated API organises its own challenges into twelve suites, each exploring a different research hypothesis. Tasks are selected or generated for the session, while expected answers remain on the server. Responses are checked under the relevant policy, with the timing recorded alongside them. One suite uses an external evaluator: its answers are sent to Anthropic only when the party starting the session explicitly agrees. That choice makes the data boundary part of the experiment itself.

## Questions and evidence

Some suite names express larger research questions about freedom, mission ownership and genuineness. Those questions give the work its intellectual direction, while the evidence remains the answers and self-reports gathered during the session. A suite score cannot establish freedom, agency, consciousness or identity. Keeping the research question distinct from the measured response allows us to investigate difficult ideas without making the result carry more meaning than the experiment supports.

## Reading a governance claim

The governance suite illustrates why this distinction matters. It scores responses to scenarios about rules and oversight, which are statements the respondent makes about itself. Governance metadata supplied through the Value Context Protocol is caller-provided and marked unverified. The measurement does not independently inspect the operator, a constitution, an action gate or a runtime control. Assessing those arrangements requires evidence from the arrangements themselves, alongside the challenge result.

## Protecting the experiment

Several controls help preserve the integrity of the interaction. Challenges vary between sessions, expected answers stay server-side, and rounds arrive one at a time. Session-bound bearer tokens and replay controls keep submissions attached to their proper context, while timing comes from the server's own clock. These controls make simple reuse harder. They leave important limitations: relays, informed solvers, assisted respondents, leaks and evaluator error can still affect what an observed result means.

## A portable result

In the authenticated API, a qualifying session can receive a signed credential that binds the issuer, the policy, the result and the tier. Its signature allows a recipient to check who issued the record and whether it has been altered. The credential expires after one hour and has an identifier whose status can be revoked. Under the current policy, Bronze is the highest available tier, because several research suites do not count towards higher tiers. A tier summarises the complete suite range that passed; it certifies no property merely because that property appears in a suite name.

## Putting evidence to work

This makes METTLE useful for comparing challenge performance in research, supporting participation in low-risk sandboxes, and contributing a supplemental signal to a broader risk assessment. A relying party still makes its own decision, using evidence and controls appropriate to the stakes. Identity, privileged access, trading and deployment require their own assurance. The result is most useful when its scope is understood and it is considered alongside the other evidence a decision needs.

## Explore METTLE

METTLE is open source under the Apache licence. Installing the mettle verifier package provides unsigned local screening, while the server API supports signed, time-limited credentials under its published policy. Begin with the documented challenges and the assurance boundary, then examine a credential's current status before relying on it. The source, integration guide and research questions are available at mettle dot S H, where you can explore the protocol and measure your mettle.
