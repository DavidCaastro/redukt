# Legal Analysis — Commercial Viability of Redukt

Analysis date: 2026-10-21. Comprehensive review of licensing, patents, regulations,
and training data risks for commercializing a compression product built on RWKV-7.

---

## Executive Verdict

**No legal blockers for commercialization.** One residual risk (training data copyright)
requires contractual mitigation but does not prevent commercial deployment.

---

## 1. RWKV-7 License: Apache 2.0

Weights source: [BlinkDL/rwkv-7-world](https://huggingface.co/BlinkDL/rwkv-7-world)
License: Apache License 2.0
Governance: [Linux Foundation (LF AI & Data)](https://lfaidata.foundation/blog/2024/02/06/rwkv-joins-lf-ai-data-as-new-incubation-project/), incubation project since Sep 2023.

### What Apache 2.0 Permits

- Commercial use without royalties
- Creation of derivative works (our product)
- Distribution of modifications under any license
- SaaS deployment without redistributing code or weights
- Sublicensing to customers
- Patent grant from all contributors

### Compliance Obligations

| Obligation | When triggered | How to comply |
|---|---|---|
| Include copy of license | If redistributing code/weights | LICENSE file in repo |
| Preserve attribution notices | Always | NOTICE file with BlinkDL credit |
| State changes made | If modifying weights/code | We use weights unmodified |
| Patent grant | Automatic | No action required |

### SaaS Model Advantage

If weights run on OUR servers (Transparent Proxy / SaaS model) and are never
distributed to the client, redistribution obligations are NOT triggered.
The client never receives the weights, model, or AZTH format internals.

Reference: [Apache 2.0 Commercial Use Guide](https://safeguard.sh/resources/blog/apache-license-version-2-0-commercial-use),
[Wiz: Apache 2.0 License Guide for Cloud Security Teams](https://www.wiz.io/academy/compliance/apache-2-license).

---

## 2. No Restrictive Clauses (Unlike Competitors)

| Model | License | Commercial restriction |
|---|---|---|
| Meta Llama 3 | Custom | >700M monthly users requires separate license |
| Google Gemma | Custom | Cannot use to train competing models |
| Mistral | Apache 2.0 | None |
| **RWKV-7** | **Apache 2.0** | **None** |

RWKV has:
- No user/volume limits
- No competing model restriction
- No acceptable use policy beyond Apache 2.0
- No restrictive CLA (Contributor License Agreement)
- No geographic restrictions (beyond general US/EU embargoes)
- No obligation to contribute changes upstream

Reference: [Open Weights vs Open Source (SoftwareSeni, 2026)](https://www.softwareseni.com/open-weights-vs-open-source-and-the-apache-2-0-licensing-shift/).

---

## 3. Training Data Risk (THE MAIN CONCERN)

### RWKV-7 World Training Data Composition

Source: [arxiv.org/pdf/2503.14456](https://arxiv.org/pdf/2503.14456) (RWKV-7 paper),
[BlinkDL/rwkv-7-world README](https://huggingface.co/BlinkDL/rwkv-7-world/blob/main/README.md).

| Dataset | License | Risk level | Notes |
|---|---|---|---|
| FineWeb-edu (HuggingFace) | Open | Low | Web scraping, standard fair use arguments |
| DCLM-baseline-1.0 | Open | Low | Curated web data |
| SlimPajama-627B (Cerebras) | Apache 2.0 | Low-Medium | Derived from RedPajama; Books3 was removed from RedPajama but may have been in earlier versions |
| The Pile (EleutherAI) | MIT (code) | **Medium** | Included Books3 (~200K copyrighted books). EleutherAI faced litigation |
| StarCoder (BigCode) | Apache 2.0 | Low | Code with permissive licenses (MIT/BSD/Apache) |
| OSCAR-2301 | Research use | Low-Medium | Web crawl, unverified provenance |

### The Books3 Problem

The Pile dataset included Books3, a collection of ~200,000 copyrighted books scraped
from shadow libraries. This led to lawsuits (Authors Guild v. various defendants).

**Key facts:**
- The books were in the training data, not in the model weights
- Model weights are a mathematical transformation, not a copy of training data
- You cannot extract original books from the weights
- RedPajama subsequently removed Books3
- No court has ruled that weights trained on copyrighted data are themselves infringing

### Risk Assessment for Our Use Case

Our product is a **compressor**, not a text generator. We use RWKV-7 as a
**probability predictor** for arithmetic coding. We do NOT:
- Generate text that could reproduce copyrighted training data
- Offer the model as a creative/generative tool
- Redistribute the training data
- Allow users to query the model for memorized content

This significantly reduces copyright risk compared to generative AI products.

Reference: [Cooley: What Enterprises Should Know Before Deploying Open-Weight AI Models (Aug 2026)](https://www.cooley.com/news/insight/2026/2026-08-12-unlocking-the-weights-what-enterprises-should-know-before-deploying-open-weight-ai-models),
[Open Weights Do Not Establish Commercial Permission (Quasa)](https://quasa.io/media/open-weight-does-not-mean-safe-for-commercial-use-check-the-license-chain).

### Mitigation Strategy

1. **SaaS model**: Do not distribute weights to clients (avoids redistribution chain)
2. **Indemnification clause**: Enterprise contracts should include IP indemnification
3. **Monitor jurisprudence**: Track Thomson Reuters v. ROSS, NYT v. OpenAI, Authors Guild cases
4. **Insurance**: Consider E&O / IP liability insurance for enterprise contracts
5. **Alternative weights**: If risk materializes, train replacement weights on clean data
   (FineWeb-edu + StarCoder only, ~3T tokens, sufficient for compression predictor)

**Practical risk level: LOW.** No tribunal has ruled that using weights trained on
copyrighted data for a non-generative purpose (compression) constitutes infringement.

---

## 4. Patent Status: ALL CLEAR

| Technology | Patent status | Risk |
|---|---|---|
| Arithmetic coding (IBM, 1978) | **Expired 2003-2006** | Zero |
| Range coding (Schindler, 1998) | **Never patented** | Zero |
| US4905297A (arithmetic encoder) | **Expired** | Zero |
| US7822283B2 (arithmetic encoding) | **Expired July 2025** | Zero |
| US9577667B2 (arithmetic encoding) | **Expired Nov 2024** | Zero |
| Context mixing (PAQ, Matt Mahoney) | **Public domain** | Zero |
| LSTM (Hochreiter & Schmidhuber, 1997) | **Academic, never patented** | Zero |
| RWKV architecture | **Apache 2.0 with patent grant** | Zero |
| Walsh-Hadamard Transform | **Public domain** | Zero |
| SafeTensors format (HuggingFace) | **Apache 2.0** | Zero |
| GGUF format (llama.cpp) | **MIT** | Zero |

**No component of our technology stack has active patent encumbrance.**

Caution: Some specialized variants of arithmetic coding patented after the original
techniques (e.g., table-based or multiply-free approximations) may still be active.
Our implementation uses the classic Schindler range coder, which predates all such patents.

Reference: [Range Coding (Wikipedia)](https://en.wikipedia.org/wiki/Range_coding),
[US7822283 (Google Patents)](https://patents.google.com/patent/US7822283/en),
[US9577667 (Google Patents)](https://patents.google.com/patent/US9577667).

---

## 5. EU AI Act: DOES NOT APPLY

The EU AI Act transparency obligations (Article 50) took effect **August 2, 2026**.

### What the AI Act Requires

- **Chatbots/agents**: Must disclose AI interaction to users
- **Synthetic content generators**: Must embed machine-readable markings
- **Open-source GPAI models**: Must comply with copyright + publish training data summary

### Why It Does Not Apply to Us

Redukt is a **data compression tool**, not:
- A chatbot or conversational AI system
- A content generation system
- A decision-making AI system
- A biometric identification system
- A high-risk AI system (Annex III)

Our product uses a neural network as a **mathematical probability predictor** for
lossless data compression. It does not interact with users, generate synthetic content,
or make decisions. It transforms bytes deterministically.

Under the AI Act classification:
- Not high-risk (not in Annex III categories)
- Not limited risk (no transparency obligations triggered)
- Not a GPAI model provider (we don't offer the model for general purposes)
- We are a **deployer** using an open-source model for a specific industrial application

### If We Add Chat/Generation Later

If future versions include generative features (chat, text generation), then
Article 50 transparency obligations WOULD apply. Keep this scope separate.

Reference: [EU AI Act High-Level Summary](https://artificialintelligenceact.eu/high-level-summary/),
[Cooley: EU AI Act Transparency Obligations](https://www.cooley.com/news/insight/2026/2026-08-03-eu-ai-act-transparency-obligations-take-effect-2-august-2026),
[Linux Foundation: What Open Source Developers Need to Know](https://linuxfoundation.eu/newsroom/ai-act-explainer).

---

## 6. Jurisdiction-Specific Notes

| Jurisdiction | Key regulation | Impact on us | Action needed |
|---|---|---|---|
| **EU** | AI Act, GDPR | AI Act: not applicable. GDPR: if processing client PII, need DPA | DPA template for EU clients |
| **Spain** | Ley Crea y Crece, RGPD, AEPD | E-invoicing format compatible. RGPD = GDPR | Same as EU |
| **US** | No federal AI law (2026). State-level varies | Minimal. Copyright risk from training data | Monitor litigation |
| **UK** | UK GDPR, Data Protection Act 2018 | Same as EU minus AI Act | DPA for UK clients |
| **Japan** | APPI (amended 2022) | Permissive on AI training data | Minimal |
| **Brazil** | LGPD | Similar to GDPR | DPA for BR clients |
| **Australia** | Privacy Act 1988, AI Ethics Principles (voluntary) | Minimal | None |
| **China** | PIPL, DSL, Algorithm Regulation | Requires local deployment review. Algorithm registration may apply | Legal counsel if entering CN market |
| **India** | DPDP Act 2023 | Permissive. Data localization for sensitive data | DPA for IN clients |
| **Canada** | PIPEDA, AIDA (proposed) | Minimal. AIDA not yet in force | Monitor AIDA |

---

## 7. Trademark and Brand Protection

### Recommended Actions

| Action | Cost | Timeline | Priority |
|---|---|---|---|
| Register "Redukt" at OEPM (Spain) | ~EUR 150 | 6-8 months | High |
| Register at EUIPO (EU-wide) | ~EUR 850 | 4-6 months | Medium |
| Register at USPTO (US) | ~$250-350 | 8-12 months | If entering US market |
| Domain: Redukt.com | ~EUR 12/year | Immediate | High |

### Name Risk

"Azathoth" is from H.P. Lovecraft's fiction (1922, public domain since 2017 in US,
1992 in EU). No trademark conflict for software/compression.

---

## 8. Certification Path (Future, If Needed)

| Certification | When needed | Cost estimate | Timeline |
|---|---|---|---|
| SOC 2 Type II | Financial/enterprise clients | EUR 15,000-50,000 | 6-12 months |
| ISO 27001 | Healthcare/government | EUR 10,000-30,000 | 6-12 months |
| HIPAA BAA | US medical imaging | Requires encryption layer | Not applicable to v1 |
| ENS (Esquema Nacional de Seguridad) | Spanish public sector | EUR 5,000-15,000 | 3-6 months |

---

## 9. Summary: What Binds Us

### Must Do (Legal Obligations)

1. Include LICENSE (Apache-2.0) in repository
2. Include NOTICE file with attribution to BlinkDL/RWKV
3. Provide DPA template for EU/UK clients processing personal data
4. Register trademark if commercializing

### Should Do (Risk Mitigation)

5. IP indemnification clause in enterprise contracts
6. E&O / IP liability insurance for enterprise tier
7. Monitor AI copyright jurisprudence quarterly
8. Maintain ability to retrain on clean data if needed

### Does NOT Bind Us

- No royalties to anyone
- No revenue sharing
- No user/volume limits
- No competing model restriction
- No copyleft (code stays proprietary if desired)
- No obligation to open source our modifications
- No geographic restrictions
- No patent licensing fees
- No EU AI Act compliance (for compression product)

---

## Sources

- [Cooley: Deploying Open-Weight AI Models (Aug 2026)](https://www.cooley.com/news/insight/2026/2026-08-12-unlocking-the-weights-what-enterprises-should-know-before-deploying-open-weight-ai-models)
- [RWKV joins Linux Foundation (Sep 2023)](https://blog.rwkv.com/p/rwkv-joins-the-linux-foundation-as)
- [LF AI & Data: RWKV Incubation](https://lfaidata.foundation/blog/2024/02/06/rwkv-joins-lf-ai-data-as-new-incubation-project/)
- [BlinkDL/rwkv-7-world (HuggingFace)](https://huggingface.co/BlinkDL/rwkv-7-world)
- [RWKV-7 Goose Paper (arxiv)](https://arxiv.org/pdf/2503.14456)
- [RWKV FAQ](https://wiki.rwkv.com/basic/FAQ.html)
- [Apache 2.0 Commercial Use (Safeguard)](https://safeguard.sh/resources/blog/apache-license-version-2-0-commercial-use)
- [Apache 2.0 License Guide (Wiz)](https://www.wiz.io/academy/compliance/apache-2-license)
- [Top 10 Apache License Questions (Mend)](https://www.mend.io/blog/top-10-apache-license-questions-answered/)
- [Open Weights vs Open Source (SoftwareSeni)](https://www.softwareseni.com/open-weights-vs-open-source-and-the-apache-2-0-licensing-shift/)
- [Open Weights Do Not Establish Commercial Permission (Quasa)](https://quasa.io/media/open-weight-does-not-mean-safe-for-commercial-use-check-the-license-chain)
- [EU AI Act High-Level Summary](https://artificialintelligenceact.eu/high-level-summary/)
- [EU AI Act Transparency (Cooley)](https://www.cooley.com/news/insight/2026/2026-08-03-eu-ai-act-transparency-obligations-take-effect-2-august-2026)
- [Linux Foundation: EU AI Act for Open Source](https://linuxfoundation.eu/newsroom/ai-act-explainer)
- [Range Coding (Wikipedia)](https://en.wikipedia.org/wiki/Range_coding)
- [US7822283 Patent - Expired](https://patents.google.com/patent/US7822283/en)
- [US9577667 Patent - Expired](https://patents.google.com/patent/US9577667)
