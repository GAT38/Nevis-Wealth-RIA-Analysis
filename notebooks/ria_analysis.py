# %% [markdown]
# # Nevis — RIA Market Analysis from SEC Form ADV data (2021 → 2026)
#
# **Author:** George Triplett · **Data:** SEC Form ADV Part 1A bulk snapshots, Sept 2021 – Sept 2026
#
# This notebook (1) parses five years of SEC Form ADV data into a clean firm-year table of every
# SEC-registered investment adviser, and (2) answers Nevis's three business questions within each
# AUM segment: **who is growing fastest, who is capacity-strained, and who looks under-staffed**.
#
# **How to read it:** the executive summary comes first; each section then gives the method, the
# code and a short commentary. Every number in the summary is printed by a cell further down.
#
# **Structure**
# 1. Setup & data acquisition
# 2. Parse & clean → firm-year panel (with a data-quality log)
# 3. The analysis-ready data table (all 12 required fields + extras)
# 4. Market segments
# 5. Q1 — Fastest-growing firms (1, 3, 5 years)
# 6. Q2 — Capacity-strained firms (AUM per advisor)
# 7. Q3 — Under-staffed firms (AUM growth vs advisor growth)
# 8. So what for Nevis — market structure & a prioritised target list
# 9. Assumptions, edge cases & limitations

# %% [markdown]
# ## Executive summary
#
# **The data.** 95,020 firm-year rows covering 21,485 SEC-registered advisers (Sept 2021 → Sept 2026),
# with all 12 required fields plus extras (client mix, succession, quality flags). 17,210 firms filed
# in 2026, managing $177.9T in regulatory AUM.
#
# **Market structure (2026).** The count is in the long tail, the money is at the top:
#
# | Segment | Firms | of which wealth managers* | Share of AUM |
# |---|---|---|---|
# | <$1B | 11,820 | 7,537 (64%) | 2% |
# | $1B–$10B | 4,019 | 1,467 (37%) | 7% |
# | $10B–$50B | 939 | 156 (17%) | 12% |
# | >$50B | 432 | 61 (14%) | 79% |
#
# \* ≥50% of AUM from individuals / high-net-worth clients, i.e. the firms whose workflow Nevis serves.
#
# **Q1 — Growth.** The median firm grew AUM ~14–15% in the last year in every segment (mostly market
# beta). The fastest *organic* growers compound at 60–135% a year, 4–10× the median. Raw leaderboards
# are polluted by re-registrations and acquisitions, so the notebook ranks organic growth separately
# and shows an M&A-inclusive view next to it (1–6% of ranked firms are flagged).
#
# **Q2 — Capacity.** Among advice-led wealth managers (platforms and robo-advisers excluded), the most
# strained firms carry 4–40× their segment's median AUM per advisor.
#
# **Q3 — Under-staffing is the norm, not the exception.** 73–78% of established firms grew AUM faster
# than advisor headcount over 1, 3 and 5 years. For the median wealth manager, AUM per advisor rose
# ~27% ($60M → $76M) while clients per advisor rose only ~9% (58 → 63): each advisor is managing more
# money per relationship, which is exactly the leverage problem AI tooling addresses.
#
# **So what for Nevis.** A composite score (growth × staffing gap × client load) over 1,349 qualifying
# firms produces a prioritised target list, with a separate enterprise (≥$10B) list led by firms such as
# Concentric, Aptus, AE Wealth and O'Shaughnessy. These are firms that are growing, stretched, and
# structurally in need of leverage per advisor.
#
# **Main caveat.** Form ADV counts "advisors" as employees doing investment advisory work (Item 5.B(1)),
# which includes analysts and portfolio staff, not only client-facing advisors. Ratios are best read
# as relative within a segment, not as absolute workloads.

# %% [markdown]
# ## 1. Setup & data acquisition
#
# **Source.** The SEC publishes a monthly snapshot of the latest Form ADV Part 1A for every
# SEC-registered adviser ([link](https://www.sec.gov/data-research/sec-markets-data/information-about-registered-investment-advisers-exempt-reporting-advisers)).
# Each row is one firm, as of its most recent filing.
#
# **Which files.** One snapshot per year, all from early September, so that the 1-, 3- and 5-year
# comparisons are exactly 12, 36 and 60 months apart:
#
# | Year | SEC file | Snapshot date |
# |---|---|---|
# | 2021 | `ia090121.zip` | 1 Sep 2021 |
# | 2022 | `ia090122.zip` | 1 Sep 2022 |
# | 2023 | `ia090623.xlsx` | 6 Sep 2023 |
# | 2024 | `ia090324.xlsx` | 3 Sep 2024 |
# | 2025 | `ia09022025.xlsx` | 2 Sep 2025 |
# | 2026 | `ia100226.zip` (contains the 1 Sep 2026 extract) | 1 Sep 2026 |
#
# `scripts/download_data.py` fetches these into `data/raw/`. Parsing logic lives in
# `src/ria_adv.py` so the notebook stays readable and the cleaning rules are testable in one place.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("..").resolve()
sys.path.insert(0, str(ROOT / "src"))
import ria_adv as adv  # noqa: E402

RAW, CACHE, OUT = ROOT / "data/raw", ROOT / "data/processed/cache", ROOT / "outputs"
OUT.mkdir(exist_ok=True)

LATEST = 2026
SEGMENTS = adv.SEGMENT_LABELS

# --- analysis thresholds (all in one place so they are easy to challenge) ---
MIN_BASE_AUM = 100e6       # growth is only ranked when the base-year AUM was >= $100M
MIN_BASE_ADVISORS = 5      # staffing-gap ranking needs >= 5 advisors in the base year
MIN_ADVISORS_CAPACITY = 3  # capacity ranking ignores 1–2 person shops (portfolio-manager boutiques)
TOP_N = 5                  # rows shown per segment in the notebook (full top-25 lists go to Excel)

pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 220)

# chart style: one hue, recessive grid
BLUE, INK, MUTED = "#2a78d6", "#0b0b0b", "#52514e"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c9c8c3", "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
                     "axes.grid": True, "grid.color": "#ecebe7", "grid.linewidth": 0.8,
                     "axes.axisbelow": True, "font.size": 10})

print("files in data/raw:")
for f in sorted(RAW.iterdir()):
    print(f"  {f.name:<18} {f.stat().st_size/1e6:6.1f} MB")

# %% [markdown]
# ## 2. Parse & clean → firm-year panel
#
# **What the parser does** (`adv.build_panel`):
#
# | Step | Rule | Why |
# |---|---|---|
# | Read | Only the ~60 Form ADV items we need, all read as text | Avoids pandas silently mangling SEC numbers, ZIPs and dollar strings |
# | Encode | CSV files are Windows-1252 | Firm names with accents otherwise fail to parse |
# | Key | `SEC#` must match `801-XXXXX`; one row per firm-year (latest filing wins) | Stable firm key across years |
# | Dollars | Strip `$`, commas → float | AUM is stored as `"1,432,805,003.00"` |
# | Dates | Mixed formats (`MM/DD/YYYY` in CSVs, ISO in Excel) | Excel and CSV years export dates differently |
# | Website | Normalise to `www.domain.com` (sub-domains reduced to the firm's main domain, e.g. `connect.creativeplanning.com` → `www.creativeplanning.com`); drop social/app links and free-hosting pages (wordpress.com, blogspot.com …); across all six years pick the domain that matches the firm's name, else the latest valid one (`website_source` records which) | The bulk file carries only one of a firm's URLs per year, often a social profile or an unrelated page |
# | City / State / Country | Main-office address; if blank (~8% of filings), the mailing address, then the books-and-records address, then the firm's address from its nearest other year (`location_source` records which). State is blank for non-US firms by design; `country` carries their location | The bulk file leaves the main-office block empty for some filers |
# | Clients | Σ Item 5.D(1) client counts over all 14 client types + Item 5.C(1) (clients with no AUM, e.g. planning-only) | Item 5 has no single "total clients" field |
# | Client edge cases | "Fewer than 5" boxes → impute 2; 5.C(1) values that are really dollar amounts → dropped | See QA log below |
# | Advisors | Item 5.B(1): employees who perform investment advisory functions | Closest Form ADV proxy for advisor headcount |
# | Wealth flag | ≥ 50% of AUM from individuals + high-net-worth individuals (Items 5.D(a)(3)+(b)(3)) | Separates wealth managers (Nevis's ICP) from institutional asset managers |
# | Platform flag | > 1,000 clients per advisor → `digital_or_platform` (robo-advisers, managed-account platforms, TAMPs) | Their "advisors" aren't client-facing; excluded from capacity rankings and targets |
# | Ratios | AUM/client, AUM/advisor, clients/advisor are NaN when the denominator is 0 | No divide-by-zero infinities in rankings |

# %%
panel, qa = adv.build_panel(RAW, cache_dir=CACHE)

qa_flags = panel.groupby("year").agg(
    clients_5c_dollar_typo=("clients_5c_invalid_flag", "sum"),
    firms_with_lt5_imputation=("clients_imputed_flag", "sum"),
    website_found=("website", lambda s: f"{s.notna().mean():.0%}"),
    website_backfilled=("website_filled_flag", "sum"),
).reset_index()
qa.merge(qa_flags, on="year")

# %% [markdown]
# **Reading the QA log.** Every snapshot parses cleanly: no invalid or duplicate SEC numbers,
# no missing AUM. About 4% of firms report $0 regulatory AUM (pure financial-planning or
# consulting advisers). They stay in the table but drop out of the growth and ratio rankings.
# A handful of filers each year typed a dollar figure into the 5.C(1) "number of clients"
# box, for example one firm reporting 2.36 *billion* clients, exactly equal to its AUM. Left in,
# those few rows would inflate the industry client total about 35×, so they are neutralised and flagged.
#
# **Sanity check: industry totals over time.** If the parse were wrong, these would jump
# around. Instead they move smoothly. (Regulatory AUM double-counts sub-advised assets, so it is
# larger than "wealth" AUM figures quoted in the press.)

# %%
totals = (panel.groupby("year")
          .agg(firms=("sec_number", "nunique"), aum_tn=("aum", lambda s: s.sum() / 1e12),
               clients_m=("clients", lambda s: s.sum() / 1e6), employees_k=("employees", lambda s: s.sum() / 1e3),
               advisors_k=("advisors", lambda s: s.sum() / 1e3))
          .round(1))
totals

# %% [markdown]
# **Spot check against a known firm.** The brief uses Beacon Pointe as its website example.
# Here it is across all six years. Note that its only website in the bulk file is a social
# profile, so `website` is blank and `social_url` carries the link. This is a known limitation of
# the bulk file (Section 9).

# %%
cols = ["year", "sec_number", "firm_name", "city", "state", "aum", "clients", "employees",
        "advisors", "filing_date", "website", "social_url"]
panel.loc[panel["firm_name"].str.contains("BEACON POINTE", na=False), cols]

# %% [markdown]
# ## 3. The analysis-ready data table
#
# One row per firm per year with the 12 required fields, plus a few extras that the business
# questions need:
#
# * `segment`: AUM band, so every question can be answered within segment
# * `wealth_manager`, `pct_aum_individuals`, `pct_aum_hnw`: who is a wealth manager vs an
#   institutional asset manager. This matters because "AUM per advisor" means something very
#   different at an index-fund manager than at an RIA serving families.
# * `clients_per_advisor`: a second capacity measure. Admin workload (the work Nevis automates)
#   scales with client relationships, not just dollars.
# * `crd`, `legal_name`, `snapshot_date`, `source_file`: traceability back to the SEC record
#
# The full panel is saved to `outputs/ria_panel_2021_2026.csv`, and the latest year with all
# growth metrics to `outputs/nevis_ria_analysis.xlsx`.

# %%
panel["segment"] = adv.assign_segment(panel["aum"])

REQUIRED = {
    "sec_number": "SEC#", "firm_name": "Firm Name", "website": "Website", "city": "City",
    "state": "State", "aum": "AUM (USD)", "clients": "Number of Clients",
    "aum_per_client": "Avg AUM per Client (USD)", "employees": "Number of Employees",
    "advisors": "Number of Advisors", "aum_per_advisor": "AUM per Advisor (USD)",
    "filing_date": "Filing Date", "year": "Year",
}
EXTRA = ["country", "location_source", "segment", "wealth_manager", "digital_or_platform", "pct_aum_individuals", "pct_aum_hnw",
         "clients_per_advisor", "website_source", "social_url", "crd", "legal_name", "sec_status", "snapshot_date", "source_file",
         "clients_imputed_flag", "clients_5c_invalid_flag", "website_filled_flag", "succession_flag"]

EXTRA_NAMES = {
    "country": "Country", "location_source": "Location Source",
    "segment": "Segment", "wealth_manager": "Wealth Manager", "digital_or_platform": "Digital or Platform",
    "pct_aum_individuals": "% AUM Individuals", "pct_aum_hnw": "% AUM HNW",
    "clients_per_advisor": "Clients per Advisor", "website_source": "Website Source",
    "social_url": "Social URL", "crd": "CRD#", "legal_name": "Legal Name", "sec_status": "SEC Status",
    "snapshot_date": "Snapshot Date", "source_file": "Source File",
    "clients_imputed_flag": "Flag: Clients Imputed", "clients_5c_invalid_flag": "Flag: Client Count Typo Removed",
    "website_filled_flag": "Flag: Website From Another Year", "succession_flag": "Flag: Succession",
}


def tidy(df: pd.DataFrame) -> pd.DataFrame:
    """Presentation rounding for exports: whole dollars and counts, 1 dp ratios, 4 dp shares/rates."""
    df = df.copy()
    whole = ["aum", "clients", "employees", "advisors", "aum_per_client", "aum_per_advisor", "crd"]
    whole += [c for c in df.columns if c.startswith(("aum_", "advisors_", "clients_")) and c.endswith("y_ago")]
    for c in whole:
        if c in df:
            df[c] = df[c].round(0).astype("Int64")
    for c in df.columns:
        if c == "clients_per_advisor":
            df[c] = df[c].round(1)
        elif c.startswith(("pct_", "aum_cagr", "advisor_cagr", "clients_cagr", "staffing_gap", "nevis_score")):
            df[c] = df[c].round(4)
    return df


table = panel[list(REQUIRED) + EXTRA].sort_values(["year", "aum"], ascending=[False, False])
tidy(table).rename(columns={**REQUIRED, **EXTRA_NAMES}).to_csv(OUT / "ria_panel_2021_2026.csv", index=False)

print(f"{len(table):,} firm-year rows · {table['sec_number'].nunique():,} distinct firms · "
      f"{len(table.columns)} columns → outputs/ria_panel_2021_2026.csv")
table[table["year"] == LATEST].head(10).rename(columns=REQUIRED).iloc[:, :13]

# %% [markdown]
# ## 4. Market segments (latest year, Sept 2026)

# %%
latest = panel[panel["year"] == LATEST]
seg = (latest.groupby("segment", observed=True)
       .agg(firms=("sec_number", "count"),
            wealth_managers=("wealth_manager", "sum"),
            aum_tn=("aum", lambda s: s.sum() / 1e12),
            median_aum_m=("aum", lambda s: s.median() / 1e6),
            median_advisors=("advisors", "median"),
            median_clients=("clients", "median")))
seg["share_of_firms"] = seg["firms"] / seg["firms"].sum()
seg["share_of_aum"] = seg["aum_tn"] / seg["aum_tn"].sum()
seg["pct_wealth_managers"] = seg["wealth_managers"] / seg["firms"]
seg.loc["Total"] = [seg["firms"].sum(), seg["wealth_managers"].sum(), seg["aum_tn"].sum(),
                    latest["aum"].median() / 1e6, latest["advisors"].median(),
                    latest["clients"].median(), 1.0, 1.0,
                    seg["wealth_managers"].sum() / seg["firms"].sum()]

fmt = seg.copy()
for c in ["firms", "wealth_managers", "median_advisors", "median_clients"]:
    fmt[c] = fmt[c].map("{:,.0f}".format)
fmt["aum_tn"] = fmt["aum_tn"].map("${:,.1f}T".format)
fmt["median_aum_m"] = fmt["median_aum_m"].map("${:,.0f}M".format)
for c in ["share_of_firms", "share_of_aum", "pct_wealth_managers"]:
    fmt[c] = fmt[c].map("{:.0%}".format)
fmt

# %%
s = seg.drop("Total")
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for ax, col, title in [(axes[0], "firms", "Number of RIAs"), (axes[1], "aum_tn", "Regulatory AUM ($T)")]:
    bars = ax.bar(s.index.astype(str), s[col], color=BLUE, width=0.6)
    ax.set_title(title, loc="left", fontsize=11)
    for b, v in zip(bars, s[col]):
        ax.annotate(f"{v:,.0f}" if col == "firms" else f"${v:,.0f}T", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=9, color=INK, xytext=(0, 2), textcoords="offset points")
    ax.grid(axis="x", visible=False)
fig.suptitle(f"RIA market by AUM segment, Sept {LATEST}", x=0.01, ha="left", fontsize=12, color=INK)
fig.tight_layout()

# %% [markdown]
# **Market structure in one line:** the market is a long tail by count and concentrated by dollars.
# About 69% of the ~17,200 RIAs manage under $1B, but the ~430 firms above $50B hold most of the
# regulatory AUM. The bigger the segment, the less "wealth management" it is: most firms under $1B
# are wealth managers, but most firms above $10B are institutional asset managers
# (funds, pensions, insurers). **For Nevis, segment by AUM *and* by client type.** The
# `wealth_manager` flag is what turns a list of 17,000 advisers into a list of real prospects.

# %% [markdown]
# ## 5. Q1 — Who are the fastest-growing firms?
#
# **Method.**
# * Growth is the **compound annual growth rate (CAGR)** of regulatory AUM between the September
#   snapshots: 2025→2026 (1y), 2023→2026 (3y) and 2021→2026 (5y). Total growth is shown alongside.
# * A firm is ranked only if it existed in the base year **and** had ≥ $100M AUM then. Otherwise
#   a firm going from $2M to $900M "grows 45,000%" and crowds out every real story.
# * Segment = **current** (2026) AUM, so this answers "who are the fastest growers among today's
#   $1–10B firms".
# * **Organic vs. structural growth.** Some AUM jumps aren't growth in the commercial sense:
#   * a new SEC registrant taking over an affiliate's book (e.g. a bank moving its wealth arm into a new entity)
#   * a fund launch or a roll-up
#
#   Two flags catch these:
#   * `step_change`: AUM more than tripled in a single year inside the window
#   * `succession_any`: the firm reported succeeding to another adviser's business (Item 4.A)
#
#   The **headline tables below rank organic growers only** (neither flag). The flagged firms are
#   shown once afterwards, and all lists are in the Excel workbook. M&A-driven firms are still
#   excellent prospects, since integration is a buying moment, but they need a different story.
# * **Market context:** median AUM growth was ~15% in the last year in every segment. A firm has
#   to beat that to be taking share.

# %%
g = adv.growth_table(panel, LATEST)


def money(x):
    if pd.isna(x):
        return ""
    return f"${x/1e9:,.1f}B" if abs(x) >= 1e9 else f"${x/1e6:,.0f}M"


def pct(x):
    return "" if pd.isna(x) else f"{x:+.0%}"


def hq(t):
    """City, State for US firms; City, Country for non-US firms."""
    region = t["state"].where(t["state"].str.len() > 0, t["country"].str.title())
    return t["city"].str.title() + ", " + region


def flag(t, h):
    return np.where(t[f"step_change_{h}y"] | t["succession_any"], "M&A / step change", "")


def top_by_segment(df, sort_col, n, ascending=False):
    return (df.sort_values(sort_col, ascending=ascending)
              .groupby("segment", observed=True, group_keys=False).head(n)
              .sort_values(["segment", sort_col], ascending=[True, ascending]))


def fastest(h, n=TOP_N, organic=True):
    d = g[(g[f"aum_{h}y_ago"] >= MIN_BASE_AUM) & g[f"aum_cagr_{h}y"].notna()]
    d = d[~d[f"step_change_{h}y"] & ~d["succession_any"]] if organic else d
    return top_by_segment(d, f"aum_cagr_{h}y", n)


def show_fastest(h, organic=True):
    t = fastest(h, organic=organic)
    return pd.DataFrame({
        "Segment": t["segment"].astype(str), "SEC#": t["sec_number"], "Firm": t["firm_name"].str.title(),
        "HQ": hq(t), f"AUM {LATEST - h}": t[f"aum_{h}y_ago"].map(money), f"AUM {LATEST}": t["aum"].map(money),
        f"CAGR {h}y": t[f"aum_cagr_{h}y"].map(pct),
        "Total growth": ((t["aum"] / t[f"aum_{h}y_ago"]) - 1).map(pct),
        "Advisors": t["advisors"].map("{:,.0f}".format),
        "Wealth mgr": t["wealth_manager"].map({True: "yes", False: ""}),
        **({} if organic else {"Flag": flag(t, h)}),
    }).reset_index(drop=True)


med = g.groupby("segment", observed=True)[["aum_cagr_1y", "aum_cagr_3y", "aum_cagr_5y"]].median()
flag_rate = pd.DataFrame({f"{h}y": g[g[f"aum_{h}y_ago"] >= MIN_BASE_AUM].groupby("segment", observed=True)
                          .apply(lambda d: (d[f"step_change_{h}y"] | d["succession_any"]).mean(), include_groups=False)
                          for h in (1, 3, 5)})
print("Median AUM CAGR by segment (the bar to beat):")
print(med.map("{:.1%}".format).to_string())
print("\nShare of ranked firms flagged M&A / step change:")
print(flag_rate.map("{:.0%}".format).to_string())

# %% [markdown]
# ### Fastest organic growers, 1 year (Sept 2025 → Sept 2026)

# %%
show_fastest(1)

# %% [markdown]
# ### Fastest organic growers, 3 years (2023 → 2026)

# %%
show_fastest(3)

# %% [markdown]
# ### Fastest organic growers, 5 years (2021 → 2026)

# %%
show_fastest(5)

# %% [markdown]
# ### For contrast: the 3-year list *including* M&A / step-change firms
#
# This is what an unfiltered ranking would show: mostly re-registrations and roll-ups
# (e.g. bank wealth arms moved into new entities, asset managers consolidating affiliates).

# %%
show_fastest(3, organic=False)

# %% [markdown]
# **Commentary.**
# * Unfiltered, the leaderboards are dominated by **structural jumps** (re-registrations, fund
#   launches, roll-ups), with "growth" of 1,000–6,000%. Ranking organic growers separately turns
#   the list into firms that are genuinely out-competing their market.
# * The top organic growers compound at roughly 60–135% a year over three years, 4–10× the
#   segment median. AUM growth like that outruns any operating model built for the firm's
#   2023 size.
# * Wealth managers figure prominently among the $1–10B organic growers (Root Financial, Buck
#   Wealth, Prosperity / EisnerAmper, ThePartners). That is the heart of Nevis's mid-market.

# %% [markdown]
# ## 6. Q2 — Which firms are capacity-strained?
#
# **Method.** "Capacity-strained" = the highest **AUM per advisor** (Item 5.F(2)(c) ÷ Item 5.B(1)).
# Three judgment calls, each visible in the code:
# 1. **Wealth managers only for the headline list.** At an institutional asset manager, $5B per
#    investment professional is normal (one index-fund PM runs billions) and says nothing about
#    strain. The all-firm version is in the Excel workbook.
# 2. **At least 3 advisors**, so that one- and two-person boutiques with a single large mandate
#    don't fill the list.
# 3. **No digital / platform advisers** (more than 1,000 clients per advisor: robo-advisers,
#    managed-account platforms, TAMPs), whose "advisors" are not client-facing.
#
# AUM per advisor is shown next to **clients per advisor**, the better proxy for admin load
# (meeting prep, follow-ups, paperwork), and the one Nevis's product actually relieves.

# %%
cap_universe = g[g["wealth_manager"] & ~g["digital_or_platform"]
                 & (g["advisors"] >= MIN_ADVISORS_CAPACITY) & (g["aum"] > 0)]
seg_med_cap = cap_universe.groupby("segment", observed=True)["aum_per_advisor"].median()


def show_capacity(df, n=TOP_N):
    t = top_by_segment(df, "aum_per_advisor", n)
    return pd.DataFrame({
        "Segment": t["segment"].astype(str), "SEC#": t["sec_number"], "Firm": t["firm_name"].str.title(),
        "HQ": hq(t), "AUM": t["aum"].map(money),
        "Advisors": t["advisors"].map("{:,.0f}".format), "AUM / advisor": t["aum_per_advisor"].map(money),
        "x segment median": (t["aum_per_advisor"] / t["segment"].map(seg_med_cap).astype(float)).map("{:.1f}x".format),
        "Clients / advisor": t["clients_per_advisor"].map("{:,.0f}".format),
        "AUM / client": t["aum_per_client"].map(money),
    }).reset_index(drop=True)


print("Median AUM per advisor, human-advised wealth managers with >= 3 advisors:")
print(seg_med_cap.map(money).to_string())
show_capacity(cap_universe)

# %% [markdown]
# **Commentary.**
# * In the $1B+ segments the outliers run roughly 4–40× their segment's median AUM per advisor.
#   The extreme end above $50B is direct-indexing specialists (Quantinno, Aperio), where software
#   does much of the work.
#   Under $1B the list is mechanically capped (a 3-advisor firm can't exceed ~$330M per head and
#   still be under $1B), so there the ranking surfaces the leanest small firms.
# * Two very different profiles sit behind a high AUM-per-advisor number:
#   * **Genuinely stretched teams:** high AUM *and* hundreds of clients per advisor. Every new
#     relationship adds admin to an advisor who is already full. These are the strongest Nevis prospects.
#   * **Ultra-high-net-worth boutiques:** a few dozen clients per advisor at $30–100M each. They are
#     not drowning in relationships, so the pitch is service depth and leverage, not volume.
#
#   The clients-per-advisor column separates the two, which is why it was added to the table.

# %% [markdown]
# ## 7. Q3 — Which RIAs appear under-staffed?
#
# **Method.** Staffing gap = AUM CAGR − advisor-headcount CAGR over 1, 3 and 5 years. A gap of +40
# points means AUM compounded 40 points a year faster than the advisor team.
#
# Guards:
# * at least 5 advisors in the base year, so going from 1 to 2 advisors (+100%) isn't treated as hiring
# * base AUM ≥ $100M
# * positive AUM growth, since a firm whose AUM fell slower than headcount is shrinking, not under-staffed
# * **organic growth only**, as in Q1. A firm that absorbed another firm's book on paper isn't
#   "under-staffed", it just re-registered.
# * **no headcount step-changes** (`advisor_step`): advisors more than halved or tripled in a single
#   year. This is the mirror image of the AUM rule. Groups often move staff between affiliated
#   registrants, e.g. two affiliates of one manager both dropping from 2,540 to 912 "advisors",
#   or Vanguard's parent entity going from 301 to 19 as staff moved to a sister registrant.
#   Left in, these would top the list without being under-staffed.
#
# **Industry-wide picture first.** The median firm's AUM per advisor and clients per advisor over
# time, for the human-advised wealth managers Nevis sells to:

# %%
wm = panel[panel["wealth_manager"] & ~panel["digital_or_platform"] & (panel["advisors"] >= MIN_ADVISORS_CAPACITY)]
trend = wm.groupby("year").agg(firms=("sec_number", "count"),
                               median_aum_per_advisor=("aum_per_advisor", "median"),
                               median_clients_per_advisor=("clients_per_advisor", "median"))
idx = trend[["median_aum_per_advisor", "median_clients_per_advisor"]] / trend.iloc[0][["median_aum_per_advisor", "median_clients_per_advisor"]] * 100

fig, ax = plt.subplots(figsize=(8, 3.6))
labels = {"median_aum_per_advisor": "AUM per advisor", "median_clients_per_advisor": "Clients per advisor"}
for col, color in zip(idx.columns, [BLUE, MUTED]):
    ax.plot(idx.index, idx[col], color=color, lw=2, marker="o", ms=5)
    ax.annotate(f"{labels[col]}  {idx[col].iloc[-1]:.0f}", (idx.index[-1], idx[col].iloc[-1]),
                xytext=(8, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
ax.axhline(100, color="#c9c8c3", lw=1)
ax.set_xticks(idx.index)
ax.set_xlim(idx.index[0] - 0.2, idx.index[-1] + 1.6)
ax.set_ylabel("Median firm, index 2021 = 100")
ax.set_title("Wealth managers: each advisor now carries more assets than in 2021", loc="left", fontsize=11)
fig.tight_layout()

print(trend.assign(median_aum_per_advisor=trend["median_aum_per_advisor"].map(money),
                   median_clients_per_advisor=trend["median_clients_per_advisor"].round(0)).to_string())

# %%
def understaffed(h, n=TOP_N, organic=True):
    d = g[(g[f"advisors_{h}y_ago"] >= MIN_BASE_ADVISORS) & (g[f"aum_{h}y_ago"] >= MIN_BASE_AUM)
          & (g[f"aum_cagr_{h}y"] > 0) & g[f"staffing_gap_{h}y"].notna()]
    d = d[~d[f"step_change_{h}y"] & ~d[f"advisor_step_{h}y"] & ~d["succession_any"]] if organic else d
    return top_by_segment(d, f"staffing_gap_{h}y", n)


def show_understaffed(h):
    t = understaffed(h)
    return pd.DataFrame({
        "Segment": t["segment"].astype(str), "SEC#": t["sec_number"], "Firm": t["firm_name"].str.title(),
        "HQ": hq(t),
        f"AUM {LATEST - h}→{LATEST}": t[f"aum_{h}y_ago"].map(money) + " → " + t["aum"].map(money),
        f"Advisors {LATEST - h}→{LATEST}": t[f"advisors_{h}y_ago"].map("{:,.0f}".format) + " → " + t["advisors"].map("{:,.0f}".format),
        "AUM CAGR": t[f"aum_cagr_{h}y"].map(pct), "Advisor CAGR": t[f"advisor_cagr_{h}y"].map(pct),
        "Gap (pts/yr)": (t[f"staffing_gap_{h}y"] * 100).map("{:+.0f}".format),
        "Wealth mgr": t["wealth_manager"].map({True: "yes", False: ""}),
    }).reset_index(drop=True)


share_gap = {h: (g.loc[(g[f"advisors_{h}y_ago"] >= MIN_BASE_ADVISORS) & (g[f"aum_{h}y_ago"] >= MIN_BASE_AUM),
                       f"staffing_gap_{h}y"] > 0).mean() for h in (1, 3, 5)}
print("Share of established firms whose AUM grew faster than advisor headcount:",
      ", ".join(f"{h}y {v:.0%}" for h, v in share_gap.items()))

# %% [markdown]
# ### Under-staffed, 1 year

# %%
show_understaffed(1)

# %% [markdown]
# ### Under-staffed, 3 years

# %%
show_understaffed(3)

# %% [markdown]
# ### Under-staffed, 5 years

# %%
show_understaffed(5)

# %% [markdown]
# **Commentary.**
# * Under-staffing is the norm, not the exception. About three-quarters of established firms grew
#   AUM faster than their advisor teams at every horizon. The median wealth manager's AUM per advisor
#   rose ~27% from 2021 to 2026 (≈$60M → $76M), while clients per advisor rose ~9%: each advisor is
#   responsible for more money *and* more relationships. Market appreciation does part
#   of this, since assets rise with markets while headcount doesn't, but the leaders above are far
#   beyond any market move.
# * The leaders doubled or tripled AUM on a flat or *shrinking* advisor team. Either they already
#   run unusually efficient operations, or the strain is landing on advisors and staff. Both are
#   good conversations for Nevis.
# * 1-year gaps are noisy, because one large mandate or market swing moves them. **3-year gaps are
#   the most decision-useful signal:** sustained, but recent.

# %% [markdown]
# ## 8. So what for Nevis
#
# **Market structure.**
# * Wealth managers are about two-thirds of RIAs under $1B, a third of the $1–10B segment and a
#   small minority above $10B. Nevis's enterprise motion ($10B+) is therefore hunting in a
#   **small, nameable pool** of a couple of hundred firms. The list below can go straight into a CRM.
# * The median wealth manager's AUM per advisor keeps rising. Advisor capacity is a structural,
#   industry-wide constraint, which is the core Nevis thesis, now with numbers.
#
# **A prioritised target list.** One score combining all three questions. Universe: human-advised
# wealth managers with ≥ $1B AUM, ≥ 5 advisors three years ago, and a full 3-year history. Each
# firm gets a percentile rank within this universe on:
# 1. 3-year AUM CAGR (momentum)
# 2. 3-year staffing gap (strain is growing)
# 3. clients per advisor (strain right now)
#
# The score is the average of the three. M&A / step-change firms are kept (integration is a strong
# buying moment) but flagged, so a strategist knows which story to lead with.

# %%
t = g[g["wealth_manager"] & ~g["digital_or_platform"] & (g["aum"] >= 1e9)
      & g["aum_cagr_3y"].notna() & g["staffing_gap_3y"].notna() & g["clients_per_advisor"].notna()
      & (g["advisors_3y_ago"] >= MIN_BASE_ADVISORS) & ~g["advisor_step_3y"]].copy()
for c in ["aum_cagr_3y", "staffing_gap_3y", "clients_per_advisor"]:
    t[f"pctl_{c}"] = t[c].rank(pct=True)
t["nevis_score"] = t[[c for c in t if c.startswith("pctl_")]].mean(axis=1)
targets = t.sort_values("nevis_score", ascending=False)

print(f"Scored universe: {len(t):,} wealth managers >= $1B "
      f"({t['segment'].value_counts().reindex(SEGMENTS[1:]).to_dict()})")
top = targets.head(25)
pd.DataFrame({
    "Rank": range(1, len(top) + 1), "SEC#": top["sec_number"], "Firm": top["firm_name"].str.title(),
    "Website / social": top["website"].fillna(top["social_url"]).fillna(""), "HQ": hq(top),
    "Segment": top["segment"].astype(str), "AUM": top["aum"].map(money),
    "AUM CAGR 3y": top["aum_cagr_3y"].map(pct), "Advisor CAGR 3y": top["advisor_cagr_3y"].map(pct),
    "Clients / advisor": top["clients_per_advisor"].map("{:,.0f}".format),
    "Score": top["nevis_score"].map("{:.2f}".format),
    "Flag": flag(top, 3),
}).reset_index(drop=True)

# %%
# Same scoring, enterprise only (>= $10B) — Nevis's Deployment Strategist territory
ent = targets[targets["aum"] >= 10e9].head(15)
pd.DataFrame({
    "Rank": range(1, len(ent) + 1), "SEC#": ent["sec_number"], "Firm": ent["firm_name"].str.title(),
    "HQ": hq(ent), "AUM": ent["aum"].map(money), "AUM CAGR 3y": ent["aum_cagr_3y"].map(pct),
    "Advisors": ent["advisors"].map("{:,.0f}".format), "Advisor CAGR 3y": ent["advisor_cagr_3y"].map(pct),
    "Clients / advisor": ent["clients_per_advisor"].map("{:,.0f}".format),
    "Score": ent["nevis_score"].map("{:.2f}".format), "Flag": flag(ent, 3),
}).reset_index(drop=True)

# %% [markdown]
# **How a Deployment Strategist would use this.** Sort the CRM by `nevis_score` within segment.
# Lead with momentum for high-growth firms ("you've grown X% a year, and your team hasn't"), and
# with capacity for high clients-per-advisor firms ("each advisor carries N relationships").
# Firms flagged for M&A get an integration-led pitch: "consolidate onto one platform as you absorb
# books".
#
# **Next steps for this analysis:**
# * enrich websites and contacts from IAPD Schedule D
# * refresh monthly as new ADV snapshots land
# * roll multi-registrant groups up to the parent brand using Item 7 affiliations
# * add a market-adjusted growth measure (firm growth minus the segment median)

# %% [markdown]
# ## 9. Assumptions, edge cases & limitations
#
# | Topic | Assumption / handling | Impact |
# |---|---|---|
# | **Timing of AUM** | Each snapshot shows a firm's AUM as of its *latest filing*. Most firms update AUM once a year, in the annual amendment due ~90 days after fiscal year-end (hence the March filing spike). | "2026 AUM" mostly means AUM at FY-end 2025. Growth compares like-for-like filings one, three and five years apart. |
# | **Regulatory AUM** | Item 5.F(2)(c), discretionary + non-discretionary | Includes some double counting (sub-advised assets). Fine for ranking, not for industry sizing. |
# | **Advisors** | Item 5.B(1), employees performing investment advisory functions *including research* | Overstates client-facing advisors at research-heavy firms. Could refine with 5.B(3) (state-registered IARs) for wealth firms. |
# | **Clients** | Σ 5.D(1) + 5.C(1). "Fewer than 5" boxes imputed as 2. Dollar typos in 5.C(1) dropped (≈5–10 firms a year). | Small firms' counts can be ±2 per category. Totals are no longer dominated by typos. |
# | **Market effect** | Growth is *not* market-adjusted. Each table shows the segment median as context. | 2021→2022 was a drawdown year, 2023→2026 a strong market. Compare firms to peers, not to zero. |
# | **Firm identity** | Firm = SEC# (801-). Re-registrations under a new SEC# and multi-entity groups (e.g. separate RIAs for one brand) are not merged. | Some "new" fast growers are reorganisations; flagged via the step-change rule. |
# | **Survivorship** | Growth requires presence in both years. Firms that deregistered or moved to state registration drop out. | Rankings describe surviving firms, which is the right population for prospecting. |
# | **Segment boundaries** | Lower bound inclusive: exactly $1B is in "$1B–$10B". | Negligible. |
# | **Websites** | The bulk file holds one URL per firm-year (often a social profile). Best domain chosen across six years by firm-name match. | ~82% of firms get a website: ~70% name-matched (high confidence), ~12% "latest" (lower confidence, flagged in `website_source`). Sub-domains are reduced to the main domain. Where a firm only ever reported a side site (e.g. Northern Trust's capitalmarketassumptions.com), that is what appears, flagged "latest". Full lists live in IAPD Schedule D 1.I. Some large managers (e.g. Vanguard, Fidelity) report only social links, so their Website is blank. `social_url` keeps company pages only; personal LinkedIn profiles are dropped. |
# | **Location** | ~8% of filings leave the main-office address blank. Filled from the mailing address, then books-and-records address, then the firm's nearest other year. | Missing city falls from ~8% to ~3% of rows; `location_source` records where each came from. Non-US firms have a blank State and a `country`. |
# | **Extreme ratios** | Firms reporting large AUM with 0–1 advisors are kept in the data table as filed. Zero denominators give blank ratios. | They are excluded from every ranking by the minimum-advisor thresholds, so they never reach a leaderboard. |
# | **Platforms** | > 1,000 clients per advisor = digital / platform adviser | A heuristic. A few large human-advised firms with big call-centre models may be caught; they stay in the data table. |
# | **Wealth-manager flag** | ≥ 50% of AUM from individuals + HNW clients | A judgment threshold. The Excel workbook keeps `pct_aum_individuals` so it can be moved. |

# %%
# ---- Export everything a reviewer or a sales team would want -------------------------------
cols_latest = (["sec_number", "crd", "firm_name", "website", "website_source", "social_url", "city", "state", "country", "location_source", "segment",
                "aum", "clients", "aum_per_client", "employees", "advisors", "aum_per_advisor",
                "clients_per_advisor", "wealth_manager", "digital_or_platform", "pct_aum_individuals", "pct_aum_hnw", "filing_date"]
               + [f"{m}_{h}y" for h in (1, 3, 5) for m in ("aum_cagr", "advisor_cagr", "staffing_gap", "step_change", "advisor_step")]
               + ["succession_any"])

all_cap = g[(g["advisors"] >= MIN_ADVISORS_CAPACITY) & (g["aum"] > 0)]
sheets = {
    "README": pd.DataFrame({"note": [
        "Source: SEC Form ADV Part 1A bulk snapshots, Sept 2021-Sept 2026 (see notebook Section 1).",
        f"Growth rankings require base-year AUM >= ${MIN_BASE_AUM/1e6:.0f}M; staffing gap requires >= {MIN_BASE_ADVISORS} base-year advisors.",
        f"Capacity rankings require >= {MIN_ADVISORS_CAPACITY} advisors; 'capacity_wealth' restricts to firms with >= 50% of AUM from individuals/HNW and excludes digital/platform advisers (>1,000 clients per advisor).",
        "'organic' sheets exclude firms flagged step_change or succession_any; 'incl_mna' sheets keep them.",
        "step_change = AUM more than tripled in a single year within the window; advisor_step = advisors more than halved or tripled in a single year; succession_any = Item 4.A succession reported.",
        "Segment = latest-year AUM band. All $ in USD."]}),
    "firms_2026": g[cols_latest].sort_values("aum", ascending=False),
    "segment_summary": seg.reset_index(),
    "industry_trend": trend.reset_index(),
    **{f"fastest_{h}y_organic": fastest(h, 25)[cols_latest] for h in (1, 3, 5)},
    **{f"fastest_{h}y_incl_mna": fastest(h, 25, organic=False)[cols_latest] for h in (1, 3, 5)},
    "capacity_wealth": top_by_segment(cap_universe, "aum_per_advisor", 25)[cols_latest],
    "capacity_all_firms": top_by_segment(all_cap, "aum_per_advisor", 25)[cols_latest],
    **{f"understaffed_{h}y": understaffed(h, 25)[cols_latest] for h in (1, 3, 5)},
    "nevis_targets": targets.head(100)[["nevis_score"] + cols_latest],
}
with pd.ExcelWriter(OUT / "nevis_ria_analysis.xlsx", engine="xlsxwriter") as xw:
    money_fmt = xw.book.add_format({"num_format": "#,##0"})
    pct_fmt = xw.book.add_format({"num_format": "0.0%"})
    for name, df in sheets.items():
        df = tidy(df)
        for c in df.select_dtypes(include=["datetimetz", "datetime"]).columns:
            df[c] = df[c].dt.date
        if "segment" in df:
            df["segment"] = df["segment"].astype(str)
        df.to_excel(xw, sheet_name=name[:31], index=False)
        ws = xw.sheets[name[:31]]
        ws.freeze_panes(1, 0)
        for i, c in enumerate(df.columns):  # readable number formats in Excel
            c = str(c)
            if c in ("aum", "aum_per_client", "aum_per_advisor", "clients", "employees", "advisors") or c.endswith("y_ago"):
                ws.set_column(i, i, 16, money_fmt)
            elif c.startswith(("pct_", "aum_cagr", "advisor_cagr", "clients_cagr", "staffing_gap")):
                ws.set_column(i, i, 12, pct_fmt)
print("wrote outputs/nevis_ria_analysis.xlsx with sheets:", ", ".join(sheets))
