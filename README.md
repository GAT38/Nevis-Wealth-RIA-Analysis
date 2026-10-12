# Nevis: RIA market analysis from SEC Form ADV (2021–2026)

George Triplett's take-home for the Nevis Deployment Strategist (Enterprise) role.

**Start with `notebooks/ria_analysis.ipynb`.** It is fully executed, with an executive summary at the top,
and every table and chart is reproducible from the raw SEC files. `outputs/nevis_ria_analysis.xlsx` has
the same results as sortable sheets for anyone who would rather filter than scroll.

## What's here

```
notebooks/ria_analysis.ipynb   executed notebook (the deliverable)
notebooks/ria_analysis.py      the same notebook as a plain-text "percent" script (diff-friendly source)
src/ria_adv.py                 parsing, cleaning, derived fields, growth/segment logic
scripts/download_data.py       fetches the six SEC snapshots into data/raw/
tools/build_notebook.py        executes the .py source top-to-bottom into the .ipynb
outputs/ria_panel_2021_2026.csv  clean firm-year table (95,020 rows, 21,485 firms)
outputs/nevis_ria_analysis.xlsx  all result tables, one sheet each, plus a README sheet
```

## Reproduce

```bash
pip install -r requirements.txt
SEC_USER_AGENT="Your Name you@example.com" python scripts/download_data.py   # ~70 MB
jupyter nbconvert --to notebook --execute --inplace notebooks/ria_analysis.ipynb
# or, without Jupyter:
python tools/build_notebook.py notebooks/ria_analysis.py notebooks/ria_analysis.ipynb
```

The first run parses the raw files (about a minute) and caches each snapshot under `data/processed/cache/`.
After that, a full run takes seconds.

## Data

SEC monthly Form ADV Part 1A snapshots of every SEC-registered adviser. I used one snapshot per year,
all from early September, so the 1-, 3- and 5-year comparisons are exactly 12, 36 and 60 months apart:
`ia090121.zip`, `ia090122.zip`, `ia090623.xlsx`, `ia090324.xlsx`, `ia09022025.xlsx`, `ia100226.zip` (Sept 2026 extract).

### Field mapping

| Required field | Form ADV source | Notes |
|---|---|---|
| SEC# | `SEC#` (801-xxxxx) | Validated format. Duplicates resolved to the latest filing |
| Firm Name | Primary Business Name | |
| Website | Website Address | Normalised to `www.domain.com` (sub-domains reduced to the main domain). Social links and free-hosting pages dropped. The firm's own domain is chosen across all six years by name or initials match |
| City / State | Main Office City / State (+ Country) | If blank, filled from the mailing address, then books-and-records address, then another year (`location_source`). Non-US firms have a blank State and a Country |
| AUM | Item 5.F(2)(c) | Regulatory AUM, total |
| Number of Clients | Σ Item 5.D(a–n)(1) + Item 5.C(1) | "Fewer than 5" imputed as 2. Typo guard on 5.C(1) (see notebook §2) |
| Avg AUM / Client | AUM ÷ clients | NaN when clients = 0 |
| Employees | Item 5.A | |
| Advisors | Item 5.B(1) | Employees performing investment advisory functions |
| AUM / Advisor | AUM ÷ advisors | NaN when advisors = 0 |
| Filing Date / Year | Latest ADV filing date / snapshot year | |

**Extras** (explained in the notebook): client-type AUM mix, `wealth_manager` (≥50% of AUM from
individuals + HNW), `digital_or_platform` (>1,000 clients per advisor), succession plan (Item 4.A),
and data-quality flags (`step_change` for M&A or re-registration, `advisor_step` for headcount
reclassifications, `website_source`, `clients_5c_invalid_flag`).

## Approach in one paragraph

Build a clean firm-year panel, then segment by 2026 AUM (<$1B, $1–10B, $10–50B, >$50B).
**Q1**
ranks AUM CAGR over 1, 3 and 5 years. The ranking is organic by default, with a separate M&A-inclusive view.
**Q2** ranks AUM per advisor among advice-led wealth managers with ≥3 advisors, against the segment
median.
**Q3** ranks the staffing gap (AUM CAGR − advisor CAGR) for established firms (≥$100M AUM and
≥5 advisors at the base year). It excludes acquisitions, headcount reclassifications and succession
events. Section 8 combines growth, staffing gap and client load into a single target score for Nevis.

## Key assumptions & limitations

- Regulatory AUM includes market moves. Growth figures are not net flows.
- Form ADV "advisors" includes non-client-facing investment staff, so ratios compare firms within a segment, rather than measure absolute workloads.
- Only SEC-registered advisers are covered. State-registered firms (mostly <$100M) are out of scope.
- Firms that de-registered drop out, so long-horizon rankings have survivorship bias.
- About 18% of firms list no usable website (ex. only social-media links). Those are kept, with `social_url` as a fallback. The SEC file holds one URL per firm-year, so where a firm only ever reported a side site, that is what appears (flagged `latest` in `website_source`).

The full list is in notebook §9.

## How I used AI

I built this using Claude for some tasks, which is how I would expect to work at Nevis.

- **What AI did:** retrieved and read the Form ADV instructions to map fields, wrote first drafts of the parsing and analysis code, and ran the notebook end to end.
- **What I did:** set the questions, made the judgement calls (what counts as organic growth, who counts as a wealth manager, the segment cut-offs, the target-score design) and checked the outputs against real firms.
- **Where AI output needed correcting:** the most useful part of the loop was catching AI-plausible but wrong results by spot-checking named firms. Each catch became a documented rule:
  - Client counts in the billions came from AUM typed into the client field, which led to the 5.C(1) guard.
  - Leaderboards were dominated by re-registrations, which led to the organic filter.
  - Robo-advisers and TAMPs showed up as "capacity-strained," which led to the platform filter.
  - Headcount reclassifications at Apollo/Aris and Vanguard looked like under-staffing, which led to the `advisor_step` filter.
