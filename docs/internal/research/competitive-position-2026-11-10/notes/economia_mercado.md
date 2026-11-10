# Economics and demand side (Nov 2026): what a saved byte is worth, and who pays for compression beyond zstd/gzip

Research date: 2026-11-10. All prices USD unless noted. "GB-mo" = per GB per month. Where an official pricing
page could not be rendered (AWS, GCS, Azure pages are JS-rendered and returned no tables), the figure comes from a
dated 2026 secondary guide and is flagged as such. Conversion used throughout: value of 1 GB permanently saved over a
retention horizon of T months at price p per GB-mo = p x T (no discounting, flat prices).

## Key Question 1 — Hot/cold object storage prices per GB-month (Nov 2026) and on-prem all-flash

### Takeaway
Hot object storage on the hyperscalers is $0.018-0.023/GB-mo (unchanged list price for years); cold/archive tiers are
$0.001-0.004/GB-mo; low-cost S3-compatibles sit at $0.006-0.008/GB-mo. Over a 7-year horizon a saved GB is worth
~$1.9 in S3 Standard, ~$0.3-0.34 in Glacier Instant/Flexible or Coldline, and ~$0.08-0.10 in Deep Archive/GCS
Archive. On-prem all-flash $/GB could not be sourced as a 2026 list price; 2026 flash component prices are rising sharply.

### Cited Findings
- S3 (US East), price list dated 2026-09-28, article 2026-10-09: Standard $0.023 (first 50 TB), $0.022 (next 450 TB),
  $0.021 (>500 TB); Standard-IA $0.0125; One Zone-IA $0.01; Glacier Instant Retrieval $0.004; Glacier Flexible
  $0.0036; Glacier Deep Archive $0.00099; Express One Zone $0.11; Intelligent-Tiering $0.023 down to $0.00099 —
  [Upstash S3 pricing 2026](https://upstash.com/blog/aws-s3-pricing-explained-2026.md)
- S3 retrieval fees per GB: Standard-IA/One Zone-IA $0.01; Glacier Instant $0.03; Glacier Flexible $0.01 standard /
  free bulk; Deep Archive $0.02 standard / $0.0025 bulk — [Upstash](https://upstash.com/blog/aws-s3-pricing-explained-2026.md)
- Official AWS page (fetched 2026-11-10, tables did not render): Deep Archive minimum storage duration 180 days;
  Standard-IA/One Zone-IA 30-day minimum and 128 KB minimum billable object; first 100 GB/month internet egress free
  across AWS — [AWS S3 pricing](https://aws.amazon.com/s3/pricing/)
- Google Cloud Storage (US, "mid-2026", article 2026-07-23): Standard $0.020; Nearline $0.010 (multi-region $0.015);
  Coldline $0.004; Archive $0.0012 (multi-region $0.0024). Retrieval: Nearline $0.01, Coldline $0.02, Archive $0.05 per
  GB — [dev.to cloud storage pricing 2026](https://dev.to/mr_manushukla/cloud-storage-pricing-in-2026-aws-s3-vs-azure-vs-gcp-compared-e54)
- Azure Blob (mid-2026, same article): Hot $0.018-0.023; Cool $0.010-0.013; Cold $0.0036-0.004; Archive $0.00099.
  Retrieval: Cool ~$0.01, Cold ~$0.03, Archive $0.02-0.10 per GB — [dev.to](https://dev.to/mr_manushukla/cloud-storage-pricing-in-2026-aws-s3-vs-azure-vs-gcp-compared-e54)
- Azure official page (fetched 2026-11-10, prices rendered as placeholders): early-deletion periods Cool 30 days, Cold
  90 days, Archive 180 days — [Azure Blob pricing](https://azure.microsoft.com/en-us/pricing/details/storage/blobs/)
- Cloudflare R2 Standard $0.015/GB-mo, zero egress, Class A/B operations billed after free tier —
  [Sliplane R2 alternatives 2026](https://sliplane.io/blog/cloudflare-r2-alternatives)
- Wasabi pay-as-you-go from $7.99/TB-mo (= $0.00799/GB-mo), no egress or API fees —
  [Sliplane Wasabi alternatives 2026](https://sliplane.io/blog/wasabi-alternatives)
- Backblaze B2 ~$6.95/TB-mo (= ~$0.007/GB-mo) per a 2026 comparison (R2 described as 2x B2's price); 90-day minimum
  for deleted objects — [Hackceleration R2 alternatives](https://hackceleration.com/es/labs/cambiar/cloudflare-r2)
- Hetzner Object Storage: EUR 6.49/month ex-VAT including 1 TB storage + 1 TB egress (= ~EUR 0.0065/GB-mo), extra
  egress EUR 1/TB; degraded service incidents reported July 2026 —
  [Sliplane B2 alternatives 2026](https://sliplane.io/blog/backblaze-b2-alternatives)
- On-prem flash, 2026: Pure Storage (now Everpure) disclosed a 70% customer price increase since January 2026 driven by
  300-900% input-cost surges; 30 TB enterprise SSD unit price rose 257% Q2 2025 -> Q1 2026 ($3,062 -> $10,950);
  QLC SSD capacity 16-22.6x HDD $/TB in Q1 2026 — [Verge.io all-flash cost 2026](https://www.verge.io/blog/storage/all-flash-array-cost-2026/);
  [Pulse/VDURA](https://www.pulse.bot/hardware/news/vdura-says-30-tb-qlc-ssd-capacity-now-costs-226x-more-than-hdd-edc9ea01-fd08-4007-84fd-6984769f2c6c/)
- Older AFA benchmark (2022, outdated): Gartner AFA $1.217/GB vs HDD/hybrid $0.248/GB —
  [Blocks & Files](https://blocksandfiles.com/2022/10/03/gartner-flash-array-revenue/)
- Backblaze's own pod cost ~$0.036/GB raw (480 TB pod), undated secondary figure —
  [LowEndBox](https://lowendbox.com/blog/the-latest-backblaze-drive-stats-report-out-and-one-makers-numbers-are-troubling/)
- Pure "Right-Size Guarantee" guarantees a data-reduction value and minimum effective capacity for six months, with free
  additional flash if missed — [Pure Storage press](https://www.purestorage.com/company/news-and-events/press/pure-storage-backs-flasharray-with-storage-industrys-most-comprehensive-guarantee-program.html)

### Inferences
- Value of 1 GB saved (p x months), 7 years (84 mo) / 10 years (120 mo):
  S3 Standard $1.93 / $2.76; GCS Standard $1.68 / $2.40; Azure Hot ~$1.5-1.9 / $2.2-2.8; R2 $1.26 / $1.80;
  S3 Standard-IA $1.05 / $1.50; GCS Nearline $0.84 / $1.20; Wasabi $0.67 / $0.96; B2 ~$0.58 / $0.83;
  Hetzner ~EUR 0.55 / 0.78; Glacier Instant $0.34 / $0.48; Coldline / Azure Cold $0.34 / $0.48;
  Glacier Flexible $0.30 / $0.43; GCS Archive $0.10 / $0.14; Deep Archive / Azure Archive $0.083 / $0.119.
- Applying Redukt's measured deltas vs brotli-11 (CLAUDE.md, 10 KB): enwik8 saves (2.352-0.961)/8 = 0.174 GB per raw
  GB; facturae (0.890-0.389)/8 = 0.063 GB; apache (0.466-0.216)/8 = 0.031 GB. On S3 Standard over 7 years this is
  worth $0.336, $0.121 and $0.060 per raw GB respectively — these reproduce the $0.33/$0.12/$0.06 thresholds in CLAUDE.md,
  so the break-even model is consistent with Nov-2026 list prices. In Deep Archive the same savings are worth
  $0.014, $0.005, $0.003 per raw GB (effectively zero vs any neural compute cost).
- Volume discounts (>500 TB at $0.021) and negotiated enterprise pricing lower the value of a saved byte further; the
  list price is an upper bound for large buyers.
- 2026 flash inflation (Everpure +70%) temporarily raises the value of a byte saved on all-flash on-prem systems; this
  is the one storage segment where $/GB is currently rising, not falling.

### Gaps
- Official AWS/GCS/Azure pricing tables could not be rendered; the figures above are from dated 2026 secondary guides
  and agree with each other but should be re-verified on the official pages or Price List API before publication.
- No 2026 list or street $/GB (raw or effective) found for Pure FlashArray, NetApp AFF or Dell PowerStore; vendors
  quote-only. Only a 2022 Gartner average ($1.217/GB AFA) and 2026 component inflation data were found.
- Backblaze B2 official price not fetched (secondary gives ~$6.95/TB; Backblaze's own page should be checked).

## Key Question 2 — Egress / transfer prices, EU Data Act, satellite / IoT / cellular link costs

### Takeaway
Internet egress is $0.05-0.12/GB on hyperscalers (one-off, roughly 4-5 months of hot storage per transfer); the EU Data
Act bans switching-related egress from 12 Jan 2027 but not operational egress. Constrained links change the picture by
orders of magnitude: Starlink maritime priority data is $2-5/GB, NB-IoT ~GBP 5/GB (2025) to EUR 17-30/GB (2019),
Iridium IMT $290-450/MB and Iridium SBD ~$1,000-3,000/MB (i.e. ~$1-3 million per GB).

### Cited Findings
- AWS internet egress (list dated 2026-09-28): first 100 GB free; next 10 TB $0.09/GB; next 40 TB $0.085; next 100 TB
  $0.07; over 150 TB $0.05. Inter-region examples: N. Virginia -> Ohio $0.01/GB, Oregon -> N. Virginia $0.02/GB —
  [Upstash](https://upstash.com/blog/aws-s3-pricing-explained-2026.md)
- GCP internet egress (mid-2026): first 100 GB free, then $0.12/GB (first 1 TB), $0.11 (next 9 TB), $0.08 (>10 TB).
  Azure: first 100 GB free, then $0.087/GB tiered down — [dev.to](https://dev.to/mr_manushukla/cloud-storage-pricing-in-2026-aws-s3-vs-azure-vs-gcp-compared-e54)
- EU Data Act: switching charges (including egress incurred when moving data out) prohibited from 12 January 2027
  (Art. 29(1)); 12 September 2027 concerns unfair-terms rules for older contracts; in-parallel/multi-cloud egress may
  still be charged "up to cost" after 2027 per a Commission FAQ as cited — [WZ-IT Data Act switching charges 2027](https://wz-it.com/en/blog/data-act-switching-charges-2027/)
- The Act does not abolish operational egress (bytes served to your own users every day) —
  [ecorpit cloud egress fees 2026](https://ecorpit.com/cloud-egress-fees-ai-inference-bill-comparison-2026/)
- AWS, Azure and Google waived egress for full exits in 2024; AWS's free exit requires Support approval and a ~90-day
  window; CloudFront, Direct Connect and Snow Family out of scope — [Hivelocity AWS egress fees](https://www.hivelocity.net/blog/aws-egress-fees-explained/)
- Starlink Global Priority 50 GB: $250/month (KVH rate sheet effective 2025-07-01; "still $250" in a later KVH note);
  overage $2/GB when opted in; data buckets 50 GB for $100 ($2/GB) plus terminal access fee —
  [KVH](https://www.kvh.com/?p=120900); [RV Mobile Internet](https://www.rvmobileinternet.com/starlink-adjusts-pricing-for-global-priority-and-local-priority-plans/)
- Starlink maritime tiers quoted by resellers: $350 for 100 GB; ~$1,000-1,150 for 1 TB — [Marinea rate sheet](https://www.marinea.fi/files/products/RS-Starlink-Global-Priority-Airtime-Rate-Sheet-Pricing.pdf)
- A Sept-2026 blog claims Starlink "Ocean Mode" rose to $6/GB on 2026-09-27; unverified and conflicts with $2/GB —
  [media-azi](https://old.media-azi.md/starlink-plans-o54y.html)
- Iridium SBD reseller overage: $1.08-2.88 per additional KB depending on plan (8 KB plan $2.88/KB, 30 KB plan
  $1.08/KB) — [Satellite Phone Store SBD 8 KB](https://satellitephonestore.com/catalog/sp/details/iridium-sbd-dynamic-shared-8-kb);
  other resellers $1.19-1.99 per 1,000 bytes and $0.79-3.15 per 1,000 bytes —
  [Ground Control SBD plans](https://www.groundcontrol.com/products/iridium/short-burst-data-range/short-burst-data-plans/)
- Iridium Messaging Transport (IMT, SBD successor): effective $0.29-0.45 per KB depending on pack; recommended above
  ~10 kB/month — [Ground Control IMT pricing](https://groundcontrol.com/products/iridium-messaging-transport-imt-pricing/)
- NB-IoT/LTE-M UK (Nov 2025, KeySIM on Vodafone/EE): GBP 0.95/month + GBP 0.005/MB PAYG (= ~GBP 5/GB) —
  [Pressat KeySIM](https://pressat.co.uk/releases/keysim-launches-095-narrowband-iot-sim-on-vodafone-and-ee-2e5debca3ecb2aa11c0249c2adfa351b)
- Deutsche Telekom/1NCE bundles (2019): EUR 9 for 300 MB, EUR 17 for 1 GB, EUR 30 for 3 GB (EUR 10-30/GB) —
  [TelecomTV](https://www.telecomtv.com/content/iot/the-un-carrier-adopts-some-classic-carrier-tariffing-for-its-nb-iot-service-16300)
- T-Mobile US NB-IoT: $6 per device per year with a 12 MB cap (= $0.50/MB) — [TelecomTV](https://www.telecomtv.com/content/iot/the-un-carrier-adopts-some-classic-carrier-tariffing-for-its-nb-iot-service-16300)
- Analysys Mason: IoT connectivity price per MB falling due to competition, lower still for large-volume deals —
  [Analysys Mason](https://www.analysysmason.com/research/content/articles/iot-connectivity-prices-rdme0/article-pdf)
- Enterprise satellite cost overview 2026 (Starlink vs OneWeb/SES/Viasat/Hughes for mining, maritime, aviation) —
  [TelecomLead](https://telecomlead.com/?p=128373) (no per-GB VSAT table extracted)

### Inferences
- Per-GB value of a saved byte on a single transfer: hyperscaler egress $0.05-0.12; Starlink priority $2-5
  (20-50x egress); NB-IoT ~$6-30; Iridium IMT ~$300k-450k; Iridium SBD ~$1-3M. For Redukt's text delta vs brotli
  (0.174 GB saved per raw GB on enwik8), a Starlink link at $2-5/GB yields $0.35-0.87 per raw GB per transfer — above
  the $0.33 S3 7-year threshold on a single hop, and far above the CPU compute cost only if throughput cost falls
  below that. At Iridium SBD rates, any compute cost is negligible versus link cost, but payloads are hundreds of bytes
  (model context cannot amortize; small-message regime).
- After 12 Jan 2027 the "exit egress" value of compression disappears in the EU; operational egress and CDN remain.
- Inter-region transfer ($0.01-0.02/GB) is too cheap to justify neural compression on its own.

### Gaps
- No official Starlink price page content was captured (page rendered without prices); Starlink figures are from
  authorized-reseller rate sheets dated mid-2025.
- Swarm (swarm.space) unreachable (DNS failure); no pricing or service status captured.
- No 2026 maritime/aviation VSAT $/GB (Inmarsat, Viasat) figure found; no 2026 4G/5G M2M per-GB rate card found.
- Primary EU Data Act text / Commission FAQ not fetched; dates come from consultancy blogs that cite Art. 29(1).

## Key Question 3 — Compute prices (vCPU-hour and GPU-hour), Nov 2026

### Takeaway
AWS compute-optimized vCPU is ~$0.045-0.052/vCPU-h on-demand and ~$0.014-0.018 spot (c7i/c7a); T4 is $0.25 spot /
$0.53 on-demand per GPU-h, L4 $0.58 / $0.80, A10G $0.53 / $1.01, H100 ~$4 spot / $6.9 on-demand on AWS and
$1.25-2.35/GPU-h on marketplaces or 1-year contracts. Plugging Redukt's measured throughputs in gives ~$40-50/GB (CPU
spot) and ~$16-35/GB (T4), confirming the 30-500x gap stated in CLAUDE.md.

### Cited Findings
- c7a.large: on-demand $0.10264/h, spot $0.035, 1-yr RI $0.068, 3-yr RI $0.045; 2 vCPU -> $0.0513 on-demand,
  $0.0175 spot per vCPU-h (region/OS filter not confirmed; assumed us-east-1 Linux) —
  [Vantage c7a.large](https://instances.vantage.sh/aws/ec2/c7a.large)
- c7i.large: on-demand $0.08925/h, spot $0.028, 1-yr $0.059, 3-yr $0.039; 2 vCPU -> $0.0446 on-demand, $0.014 spot
  per vCPU-h — [Vantage c7i.large](https://instances.vantage.sh/aws/ec2/c7i.large)
- g4dn.xlarge (1x T4, 4 vCPU): on-demand $0.526/h, spot $0.247 — [Vantage g4dn.xlarge](https://instances.vantage.sh/aws/ec2/g4dn.xlarge)
- g6.xlarge (1x L4 24 GB): on-demand $0.8048/h, spot $0.582 — [Vantage g6.xlarge](https://instances.vantage.sh/aws/ec2/g6.xlarge)
- g5.xlarge (1x A10G 24 GB): on-demand $1.006/h, spot $0.525 — [Vantage g5.xlarge](https://instances.vantage.sh/aws/ec2/g5.xlarge)
- p5.48xlarge (8x H100): on-demand $55.04/h, spot $31.934 -> $6.88 / $3.99 per GPU-h —
  [Vantage p5.48xlarge](https://instances.vantage.sh/aws/ec2/p5.48xlarge)
- H100 market (Oct 2026): on-demand $2.14 (Vast.ai) to $12.29 (Azure); median ~$3.39 across 53 providers; cheapest
  spot $1.25/h (May 2026); Nebius preemptible $0.79 vs $4.50 on-demand —
  [Neurohive cloud GPU pricing 2026](https://neurohive.io/en/guides/cloud-gpu-pricing/); [fal.ai H100 price](https://fal.ai/learn/serverless/nvidia-h100-price)
- SemiAnalysis (via Akash, Sept 2026): 1-year H100 contract price rose ~40% from $1.70/GPU-h (Oct 2025) to $2.35
  (Mar 2026); AWS raised H100 capacity-block prices several times in 2026 after a ~44% cut in June 2025 —
  [Akash H100 rental price 2026](https://akash.network/the-bid/h100-rental-price-2026-cost-per-hour/)
- T4 from $0.29/h at some providers; Jarvislabs L4 on-demand $0.44/h — [Neurohive](https://neurohive.io/en/guides/cloud-gpu-pricing/); [Jarvislabs](https://docs.jarvislabs.ai/blog/tags/l-4)
- Azure D64ads v5 spot ~$0.6958/h eastus (= ~$0.011/vCPU-h spot; on-demand not shown, date not stated) —
  [Holori D64ads v5](https://calculator.holori.com/azure/vm/standard-d64ads-v5/eastus)
- GCP C3/C4 Spot discounts "up to 91%" off on-demand (archived Jan-2025 page) — [GCP compute pricing (archive)](https://web.lumintu.workers.dev/read-https-web.archive.org/web/20250126095827/https://cloud.google.com/compute/all-pricing)

### Inferences
- Redukt CPU lockstep at ~1,180 B/s on 12 vCPU (CLAUDE.md) = 2,825 vCPU-h per raw GB -> $39.6/GB (c7i spot $0.014),
  $49.4/GB (c7a spot $0.0175), $126-145/GB on-demand. Redukt GPU T4 N=128 at ~4.2 KB/s = 66.1 GPU-h per GB ->
  $16.3/GB (g4dn spot) / $34.8/GB (on-demand); at a $0.29/h T4 marketplace rate, $19/GB. These bracket the
  CLAUDE.md figures ($50-60 CPU spot, $9-17 GPU) — the lower GPU figure requires cheaper-than-AWS T4 rental.
- Break-even throughput at the S3 7-year hot-text threshold ($0.336/raw GB): on c7i spot, 2,825 x (0.336/39.6) ~ 24
  vCPU-h per GB -> ~11.6 KB/s per vCPU (~140 KB/s on 12 vCPU); on T4 spot, 1.36 GPU-h per GB -> ~204 KB/s per T4;
  on H100 at $2/GPU-h, 0.168 GPU-h per GB -> ~1.65 MB/s per H100. For logs ($0.06 threshold) multiply required
  throughput by ~5.6.
- Spot/marketplace, not on-demand, is the relevant price for batch compression; AWS spot is 3-3.6x cheaper than
  on-demand on CPU but only ~1.4-2x on T4/L4/A10G.

### Gaps
- No GCP c3/c4 or Azure Dv5 on-demand/spot per-vCPU figures dated 2026 found; only an Azure spot number from an
  undated aggregator and a GCP "up to 91%" statement from an archived 2025 page.
- Vantage pages do not expose region/OS of the shown price; assumed us-east-1 Linux.
- RunPod/Lambda/Vast.ai list prices for T4/L4/A10G not individually captured; only aggregator ranges.

## Key Question 4 — Regulated retention: e-invoicing mandates (years, volumes, timelines) and log retention rules

### Takeaway
EU e-invoice archives are typically 10 years (Italy, Poland, France), Mexico 5 years, Spain 4 years (local copy only
in non-VeriFactu mode); Italy processes ~2.4 billion e-invoices/year and Mexico ~10.3 billion CFDI/year, but the
implied raw XML volume is only tens of TB per country per year. Log retention mandates are 5-7 years (SOX 7, HIPAA 6,
SEC 17a-4 3-6, MiFID II 5-7); 2026 observability-market estimates range $3-12 billion depending on scope.

### Cited Findings
- Italy FatturaPA: suppliers and buyers must archive for at least 10 years from issuance; SdI itself retains copies 15
  years — [Comarch archive compliance](https://www.comarch.com/trade-and-services/data-management/news/archive-compliance-how-to-store-e-invoices-for-1030-years/);
  [vatupdate](https://www.vatupdate.com/2026/06/09/archive-compliance-how-to-store-e-invoices-for-10-30-years/)
- Italy volume: Sogei Integrated Report 2025 cites over 2.4 billion e-invoices acquired in 2025 (secondary citation of
  the PDF) — [Tailride statistiche fatturazione 2026](https://tailride.so/it/blog/statistiche-fatturazione-elettronica-2026)
- Poland KSeF: every invoice archived centrally for 10 years from end of year of issuance; longer statutory periods
  require separate taxpayer archiving; mandatory for large taxpayers Feb 2026, most others Apr 2026, micro Jan 2027 —
  [vatupdate KSeF archiving](https://www.vatupdate.com/2025/11/15/archiving-e-invoices-on-polands-ksef-portal-under-the-2026-e-invoicing-mandate/);
  [Lleverage EU e-invoicing 2026](https://www.lleverage.ai/blog/eu-e-invoicing-manufacturers-2026-guide)
- Spain: VeriFactu-mode taxpayers not strictly required to keep local copies (AEAT holds records); non-VeriFactu mode
  must retain tamper-proof records locally for four years; VeriFactu postponed twice, now January 2027 —
  [VAT IT e-invoicing retention](https://vatit.com/blog/e-invoicing-data-retention/); [vatupdate Spain briefing](https://www.vatupdate.com/2026/03/25/e-invoicing-in-spain-a-briefing-document)
- Mexico CFDI: minimum 5 years retention; Rule 2.9.21 gives SAT permanent real-time database access —
  [VAT IT](https://vatit.com/blog/e-invoicing-data-retention/); [e-invoicing dashboard](https://einvoicing-dashboard.pages.dev/)
- Mexico volume: SAT reported 10,323 million CFDI issued in 2023 (327 per second) —
  [El Contribuyente](https://www.elcontribuyente.mx/2024/03/sat-rompe-record-327-facturas-por-segundo-durante-2023/)
- France: archiving 10 years in readable format accessible to the tax authority; rollout September 2026 (postponed
  from July 2024), later phases into 2027 — [e-invoicing dashboard](https://einvoicing-dashboard.pages.dev/);
  [Fiscal Requirements France archiving](https://www.fiscal-requirements.com/news/5556-frances-2026-einvoicing-reform-archiving-rules)
- Germany: receipt mandatory since 1 Jan 2025; issuance phased 2026-2027 — [Lleverage](https://www.lleverage.ai/blog/eu-e-invoicing-manufacturers-2026-guide)
- EU ViDA: full intra-EU B2B e-invoicing/digital reporting by 2030 — [e-invoicing dashboard](https://einvoicing-dashboard.pages.dev/)
- General EU retention ~8 years minimum per one guide; Italy and France 10 — [OriginStamp retention by country](https://originstamp.com/en/blog/reader/e-invoice-retention-periods-by-country)
- SOX: 7 years; HIPAA: 6 years (applies to policies/documentation evidencing compliance); SEC 17a-4: 6-year WORM
  retention per one source, 3-6 years per another (conflict); MiFID II: 5-year minimum —
  [TechJack retention requirements](https://techjacksolutions.com/security/it-log-and-record-retention-requirements/);
  [Aptible HIPAA audit logs](https://www.aptible.com/hipaa/audit-log-retention); [Velt audit trail guide](https://velt.dev/blog/financial-audit-trail-compliance-guide);
  [SigNoz log retention](https://signoz.io/guides/log-retention/)
- Observability market 2026: $3.40B (Coherent), $3.35B 2026 -> $6.93B 2031 (Mordor), $3.2B 2026 -> $10.4B 2034
  (Research and Markets), $11.91B 2026 -> $22.99B 2031 (MarketsandMarkets); Gartner data-observability (narrow)
  $346.4M in 2024 — [MarketsandMarkets](https://www.marketsandmarkets.com/blog/ICT/observability-tools-and-platforms-market);
  [Mordor](https://www.giiresearch.com/report/moi2035009-observability-market-share-analysis-industry.html);
  [Research and Markets](https://www.researchandmarkets.com/reports/6243238/observability-market-outlook-market-share)

### Inferences
- Assuming 5-20 KB per XML invoice (assumption, not sourced): Italy's 2.4B invoices/year = ~12-48 TB raw/year;
  Mexico's 10.3B = ~50-200 TB raw/year. A full 10-year Italian archive is therefore ~0.1-0.5 PB raw. With Redukt's
  facturae delta vs brotli (0.063 GB saved per raw GB) on S3 Standard for 10 years ($2.76/GB), each annual Italian
  cohort is worth only ~$2-8k in storage savings nationally. E-invoice XML is a compliance niche, not a volume
  market; its value is driven by mandated retention length, not bytes.
- Spain's VeriFactu removes the local archive obligation for VeriFactu-mode taxpayers, shrinking the Spanish
  e-invoice archive segment rather than growing it; Poland's KSeF central 10-year archive does the same.
- Log retention (5-7 years) multiplies the value of a saved log byte: a mid-size Datadog customer at 100 GB/day
  (see Q5) accumulates ~36 TB/year, i.e. 250 TB over a 7-year SOX horizon — two to three orders of magnitude more
  than a national e-invoice archive per customer.

### Gaps
- No primary tax-authority or Commission text fetched for any mandate; all retention periods are from compliance
  vendor blogs/trackers (consistent with each other for Italy/Poland/France at 10 years).
- No CFDI volume for 2024/2025 (only 2023); no KSeF annual volume; no official Sogei PDF accessed.
- Germany 2028 milestone not confirmed in results; ViDA archive period not found.
- MiFID II 5-year (extendable to 7) found in only one secondary source; SEC 17a-4 figures conflict (3-6 vs 6).
- No Gartner/IDC log-management-specific spend figure found; observability market estimates differ 4x by scope.
- Average XML e-invoice size (KB) not sourced; the volume inference above depends on it.

## Key Question 5 — Do buyers pay per-GB for compression itself? Implied willingness to pay (WTP)

### Takeaway
Observability vendors monetize bytes (Datadog $0.10/GB ingested, Elastic $0.017-0.019/GB-mo retained, Splunk list
~$150-200/GB/day), and compression vendors exist with outcome-based pricing (Granica takes a percentage of savings;
Pure guarantees data-reduction ratios; YScope raised $3.9M on 169x log compression at Uber). But end customers are
billed on uncompressed ingest/events, so the buyer who captures compression value is the platform operator (COGS),
not the end customer; no public per-GB price for "compression as a product" was found.

### Cited Findings
- Datadog Log Management (official, fetched 2026-11-10): ingest from $0.10 per ingested/scanned GB; Standard Indexing
  $1.70 per million events (15-day retention, annual) or $2.55 on-demand; Flex Storage from $0.05 per million events
  stored/month (annual) or $0.075 on-demand; Flex Logs Starter $0.60 / $0.90; Flex Compute "contact us"; archive
  forwarding to S3/GCS/Azure included; rehydration $0.10 per compressed GB scanned; log forwarding to custom
  destinations $0.25/GB outbound; Flex minimum 30 days; volume discounts from 3B+ events/month —
  [Datadog pricing](https://www.datadoghq.com/pricing/?product=log-management)
- Elastic Serverless Observability (official): ingest from $0.07/GB (Logs Essentials) / $0.09/GB (Complete);
  retention $0.017 / $0.019 per GB retained per month; egress 50 GB free then $0.05/GB; new metrics pricing effective
  2026-07-01 — [Elastic pricing](https://www.elastic.co/pricing/serverless-observability)
- Splunk: official pages say unit price per GB drops >50% from 1 GB/day to 100 GB/day; third parties quote list
  $150-200 per GB/day on the current rate card or $1,800-2,700 per GB/day annually; discounts of 30-50% below list for
  >500 GB/day commitments — [Splunk ingest pricing](https://www.splunk.com/en_us/products/pricing/ingest-pricing.html);
  [Ogma Splunk licensing 2026](https://ogma.in/blog/the-complete-guide-to-splunk-licensing-in-2026-ingest-vs-workload-pricing-explained);
  [Redress Splunk negotiation](https://redresscompliance.com/splunk-renewal-negotiation.html)
- Cost at scale: 100 GB/day = $300/month Datadog ingestion-only vs ~$20,000/month with full indexing; 500 GB/month =
  $43.50 on Elastic Serverless vs $1,360.72 on Datadog with standard indexing (Sept 2026) —
  [monitoringcost 100 GB](https://monitoringcost.com/log-management-cost-100gb); [livetail log management costs](https://livetail.pages.dev/blog/what-log-management-actually-costs)
- Granica Crunch: outcome-based pricing, customer pays a percentage of the savings generated, no upfront cost; reduces
  Parquet/columnar files 15-60% losslessly; percentage not published — [Granica Crunch FAQ](https://docs.granica.ai/faq/crunch);
  [How crunching works](https://docs.granica.ai/how-crunching-works/)
- YScope (CLP): University of Toronto spinout, $3.9M seed; CLP open source; powers Uber's production logging; Uber
  reported 169x compression on its logs — [BetaKit](https://betakit.com/yscope-closes-3-9-million-usd-to-track-computer-activity-amid-the-rise-of-agentic-ai/);
  [InfoQ Uber CLP](https://www.infoq.com/news/2022/11/uber-compressed-log-processor); no pricing published —
  [YScope docs](https://docs.yscope.com/clp/main)
- Pure Storage: "Data Reduction – Guaranteed" / Right-Size Guarantee: guaranteed data-reduction value and minimum
  effective capacity over six months, free flash added if missed — [Pure press release](https://www.purestorage.com/company/news-and-events/press/pure-storage-backs-flasharray-with-storage-industrys-most-comprehensive-guarantee-program.html);
  [Pure efficiency guarantee infographic](https://www.purestorage.com/content/dam/pdf/en/infographics/ig-cloud-efficiency-guarantee.pdf)
- Sumo Logic / Edge Delta critiques of Flex Logs: storage billed per event, compute separate —
  [Sumo Logic on Flex Logs](https://www.sumologic.com/blog/should-know-about-datadog-flex-logs.md); [Edge Delta](https://edgedelta.com/company/blog/the-pros-and-cons-of-datadog-flex-logs)

### Inferences
- Implied value of a GB of logs to an end customer on Datadog Flex: at ~4M events/GB (assumption used by one
  calculator), Flex storage = $0.20/GB-mo (annual) — 9x S3 Standard. But it is billed per event, so client-side
  compression does not reduce the bill; only Datadog's own COGS benefits. The same holds for the $0.10/GB ingest
  (billed on uncompressed GB) and Splunk ingest licenses.
- Where end customers DO capture compression savings directly: Datadog rehydration at $0.10 per compressed GB; Elastic
  retention at $0.017-0.019 per GB retained (if billed on stored, compressed size — not confirmed); customer-owned
  archives in S3/GCS (Datadog archive forwarding is included, so the customer pays only object storage).
- Granica's "% of savings" model proves WTP exists for lossless storage compression, but only when compute cost is a
  small fraction of the saving (Granica's method is lightweight columnar re-encoding, not neural). YScope's funding
  and Uber adoption show WTP for ~100x log compression at near-zero marginal compute — i.e. the log segment already
  has a cheap structured-log competitor that neural CM must beat on ratio AND cost.
- The buyer segment that demonstrably pays per-GB is the observability platform (Datadog/Splunk/Elastic) as a COGS
  optimizer, and storage-array vendors (Pure) as a capacity-guarantee lever — not the log-emitting enterprise.

### Gaps
- Granica's savings-share percentage not public; no CLP/YScope price list; no Pure/NetApp effective-capacity $/GB
  tied to compression guarantees.
- Whether Elastic retention is billed on compressed-on-disk or raw GB not confirmed from the pricing page.
- No published case of an enterprise paying a premium per GB for a stronger-than-zstd compressor on its own data was
  found; absence of evidence noted.

## Key Question 6 — Trend: storage $/GB decline vs GPU/inference $/token decline

### Takeaway
LLM inference cost at constant capability has fallen ~5-10x per year (2023-2026), while HDD $/GB fell ~56% over
2017-2022 (~15%/yr) and has flattened; in 2025-2026 HDD and SSD prices rose (HDD +46% since Sept 2025 per one
low-quality source; 30 TB SSD +257%). Compute per unit of model work falls far faster than storage prices, so the
economics of neural compression improve over time — but GPU rental prices themselves rebounded ~40% from Oct 2025 to
Mar 2026, so the gain comes from efficiency per GPU-hour, not from cheaper hours.

### Cited Findings
- a16z "LLMflation": cost for an LLM of equivalent performance falling about 10x per year —
  [a16z](https://a16z.com/llmflation-llm-inference-cost/)
- NeurIPS workshop paper "The Price of Progress": ~5-10x per year decline at fixed benchmark performance; fastest
  drops concentrated in 2024; reasoning models generate more tokens so per-token price understates spend —
  [arXiv 2511.23455](https://arxiv.org/html/2511.23455v1)
- Epoch-based figure cited: 10-1,000x per year depending on performance level —
  [LongTermWiki summary](https://www.longtermwiki.com/resources/2255e8e1cf26d155)
- Backblaze: 56% decrease in cost per GB between 2017 and November 2022; "steepness of the price decline curve is
  lessening"; Backblaze predicted 1 cent/GB by mid-2025 — [Blocks & Files](https://blocksandfiles.com/?p=45480);
  [TechSpot](https://www.techspot.com/news/96972-hard-drives-expected-reach-penny-gigabyte-mid-2025.html)
- Historical HDD $/GB decline ~20%/yr (1990s) and ~30%/yr (2000s) — [National Academies](https://www.nationalacademies.org/read/27445/chapter/3)
- "Storage prices have sort of stopped dropping" (independent blog) — [petercai.com](https://petercai.com/storage-prices-have-sort-of-stopped-dropping/)
- HDD prices up an average 46% since September 2025 (SEO-style site, low reliability) —
  [storagediskprices HDD 2020-2026](https://storagediskprices.com/hdd-prices-2020-2026/)
- 30 TB enterprise SSD +257% Q2 2025 -> Q1 2026; Everpure +70% customer prices since Jan 2026 —
  [Verge.io](https://www.verge.io/blog/storage/all-flash-array-cost-2026/)
- H100: rental peaked $7-10/GPU-h in 2023-24; 2026 on-demand $2-12 (median $3.39); 1-year contracts rose from $1.70
  (Oct 2025) to $2.35 (Mar 2026), +40% (SemiAnalysis) — [Neurohive](https://neurohive.io/en/guides/cloud-gpu-pricing/);
  [Akash](https://akash.network/the-bid/h100-rental-price-2026-cost-per-hour/)
- Backblaze 2025 Drive Stats: cost per gigabyte "has declined steadily over the years", smaller drives harder to
  source — [StorageNewsletter 2026-02-18](https://www.storagenewsletter.com/2026/02/18/backblaze-publishes-2025-drive-stats-report-13-years-of-data-show-growing-healthier-drive-fleet/)

### Inferences
- Cloud object-storage list prices (S3 Standard $0.023) have not moved in the 2026 sources, consistent with a flat
  storage-price curve; the value of a saved byte in hot storage is therefore roughly constant in nominal terms, while
  the compute to produce the saving (per fixed model quality) falls 5-10x/yr. Ratio moves ~5-10x/yr in favour of
  neural compression, provided the compressor captures efficiency gains (kernels, quantization, batching) rather than
  relying on falling GPU-hour prices, which rose in 2026.
- The 2025-2026 memory/flash shortage is a short-term tailwind: it raises on-prem $/GB (Everpure +70%) while
  compression compute keeps getting cheaper per unit of work.
- Caveat: the 5-10x/yr inference decline measures price at fixed benchmark quality across model generations, not the
  cost of running one fixed model; Redukt's fixed 130M-parameter backbone benefits only from hardware/kernel
  efficiency (slower), unless it adopts smaller/faster backbones (R-105 path).

### Gaps
- No single reliable 2020-2026 HDD $/GB series found (Backblaze's own 2023-2025 $/GB chart not located; only the 2022
  update and a low-quality 2026 site).
- No source quantifying $/GB trend for cloud object storage list prices over 2020-2026 (the S3 Standard price history
  is well known but was not captured from a source in this research).
- GPU $/hour time series 2023-2026 is only available piecemeal (peak $7-10 unsourced beyond aggregator blogs).
