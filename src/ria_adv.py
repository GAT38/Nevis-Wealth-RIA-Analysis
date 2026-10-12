"""
ria_adv.py — load, clean and reshape SEC Form ADV (Part 1A) bulk snapshots
into an analysis-ready RIA firm-year panel.

Source: SEC "Information About Registered Investment Advisers and Exempt
Reporting Advisers" monthly snapshots
(https://www.sec.gov/data-research/sec-markets-data/information-about-registered-investment-advisers-exempt-reporting-advisers).
Each snapshot has one row per SEC-registered adviser, reflecting that firm's
most recent Form ADV filing as of the snapshot date.

Design choices (see README for the full list):
* One snapshot per year, all taken at the start of September, so 1/3/5-year
  comparisons are exactly 12/36/60 months apart.
* Firm key = SEC file number (801-XXXXX). It is stable across years and is
  the identifier Nevis asked for.
* Number of clients = sum of Item 5.D(1) client counts across all 14 client
  types + Item 5.C(1) (advisory clients with no regulatory AUM, e.g. pure
  financial-planning clients). Categories where the filer ticked
  "fewer than 5 clients" (5.D(2)) contribute an imputed 2 clients (midpoint
  of 1-4) and are flagged.
* Advisors = Item 5.B(1): employees who perform investment advisory
  functions (including research). This is the closest Form ADV field to
  "advisor headcount"; it slightly overstates client-facing advisors at
  research-heavy firms.
"""
from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Snapshot catalogue
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Snapshot:
    year: int            # analysis year label
    file: str            # file name in data/raw
    as_of: str           # snapshot date (YYYY-MM-DD) as published by the SEC


SNAPSHOTS = [
    Snapshot(2021, "ia090121.zip", "2021-09-01"),
    Snapshot(2022, "ia090122.zip", "2022-09-01"),
    Snapshot(2023, "ia090623.xlsx", "2023-09-06"),
    Snapshot(2024, "ia090324.xlsx", "2024-09-03"),
    Snapshot(2025, "ia09022025.xlsx", "2025-09-02"),
    # The file published on the SEC page as "October 2026" contains the
    # 1 September 2026 extract (ia09012026.csv) — we use it as the latest year.
    Snapshot(2026, "ia100226.zip", "2026-09-01"),
]

# Base URL pieces, for scripts/download_data.py
SEC_URLS = {
    "ia090121.zip": "https://www.sec.gov/files/investment/data/information-about-registered-investment-advisers-and-exempt-reporting-advisers/ia090121.zip",
    "ia090122.zip": "https://www.sec.gov/files/investment/data/information-about-registered-investment-advisers-and-exempt-reporting-advisers/ia090122.zip",
    "ia090623.xlsx": "https://www.sec.gov/files/investment/data/information-about-registered-investment-advisers-exempt-reporting-advisers/ia090623.xlsx",
    "ia090324.xlsx": "https://www.sec.gov/files/investment/data/information-about-registered-investment-advisers-exempt-reporting-advisers/ia090324.xlsx",
    "ia09022025.xlsx": "https://www.sec.gov/files/investment/data/information-about-registered-investment-advisers-exempt-reporting-advisers/ia09022025.xlsx",
    "ia100226.zip": "https://www.sec.gov/files/investment/data/other/information-about-registered-investment-advisers-exempt-reporting-advisers/ia100226.zip",
}

# --------------------------------------------------------------------------
# Column map: raw Form ADV item -> clean name
# --------------------------------------------------------------------------

CLIENT_TYPES = list("abcdefghijklmn")  # Item 5.D categories (a)–(n)

BASE_COLS = {
    "SEC#": "sec_number",
    "Organization CRD#": "crd",
    "Primary Business Name": "firm_name",
    "Legal Name": "legal_name",
    "Main Office City": "city",
    "Main Office State": "state",
    "Main Office Country": "country",
    "Website Address": "website_raw",
    "SEC Current Status": "sec_status",
    "Latest ADV Filing Date": "filing_date",
    "5A": "employees",
    "5B(1)": "advisors",
    "5C(1)": "clients_no_raum",
    "5C(1)-If more than 100, how many": "clients_no_raum_over100",
    "5F(2)(a)": "aum_discretionary",
    "5F(2)(b)": "aum_non_discretionary",
    "5F(2)(c)": "aum",
    "5F(2)(f)": "accounts",
    "2A(1)": "reg_large_adviser",       # registered because RAUM >= $100M
    "4A": "succession_flag",            # Item 4: succeeded to another adviser's business
}
# 5.D: (1) number of clients, (2) "fewer than 5" checkbox, (3) RAUM, per type
for t in CLIENT_TYPES:
    BASE_COLS[f"5D({t})(1)"] = f"clients_{t}"
    BASE_COLS[f"5D({t})(2)"] = f"lt5_{t}"
    BASE_COLS[f"5D({t})(3)"] = f"raum_{t}"

NUMERIC = (
    ["employees", "advisors", "clients_no_raum", "clients_no_raum_over100",
     "aum_discretionary", "aum_non_discretionary", "aum", "accounts"]
    + [f"clients_{t}" for t in CLIENT_TYPES]
    + [f"raum_{t}" for t in CLIENT_TYPES]
)

# Domains that are social profiles, not a firm website
SOCIAL_DOMAINS = (
    "linkedin.com", "facebook.com", "twitter.com", "x.com", "instagram.com",
    "youtube.com", "youtu.be", "tiktok.com", "vimeo.com", "pinterest.com",
    "threads.net", "yelp.com", "google.com", "douyin.com", "reddit.com", "weibo.com",
    "wechat.com", "bit.ly", "linktr.ee", "apple.com", "spotify.com", "medium.com",
    "substack.com", "glassdoor.com", "indeed.com", "wix.com", "squarespace.com",
)

# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def _read_raw(path: Path) -> pd.DataFrame:
    """Read one SEC snapshot (zip containing a CSV, or an .xlsx)."""
    usecols = lambda c: c in BASE_COLS  # noqa: E731 — only keep what we need
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            csv_name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
            with z.open(csv_name) as fh:
                # SEC files are Windows-1252 encoded
                return pd.read_csv(fh, encoding="cp1252", usecols=usecols,
                                   dtype=str, keep_default_na=False)
    return pd.read_excel(path, engine="openpyxl", usecols=usecols, dtype=str,
                         keep_default_na=False)


def _to_number(s: pd.Series) -> pd.Series:
    """'1,432,805,003.00' -> 1432805003.0 ; '' -> NaN."""
    return pd.to_numeric(s.astype(str).str.replace(r"[,$\s]", "", regex=True)
                         .replace("", np.nan), errors="coerce")


def clean_website(url: str) -> str | float:
    """Normalise a Form ADV website to 'www.domain.com' form.

    * strips scheme, path, query, port and trailing dots
    * drops social-media profiles (not a firm website)
    * adds 'www.' only to bare registrable domains (e.g. beaconpointe.com),
      leaving genuine sub-domains (e.g. advisors.ubs.com) untouched
    """
    if not isinstance(url, str) or not url.strip():
        return np.nan
    u = url.strip().lower()
    u = re.sub(r"^[a-z]+://", "", u)          # scheme
    u = re.split(r"[/?#\s]", u, maxsplit=1)[0]  # path / query
    u = u.split(":")[0].strip(".")            # port, stray dots
    if not u or "." not in u:
        return np.nan
    host = u[4:] if u.startswith("www.") else u
    if any(host == d or host.endswith("." + d) for d in SOCIAL_DOMAINS):
        return np.nan
    labels = host.split(".")
    # two labels (beaconpointe.com) or a 2-letter-country second-level domain
    # (firm.co.uk) count as a bare domain
    bare = len(labels) == 2 or (len(labels) == 3 and labels[-2] in {"co", "com", "org", "net"} and len(labels[-1]) == 2)
    return f"www.{host}" if bare else host


_NAME_STOPWORDS = {
    "llc", "inc", "lp", "llp", "ltd", "limited", "corp", "corporation", "co", "company",
    "the", "and", "of", "group", "partners", "advisors", "advisers", "advisory",
    "management", "capital", "investment", "investments", "financial", "wealth",
    "asset", "assets", "services", "associates", "fund", "funds", "trust", "us", "na",
}


def _name_tokens(name: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", str(name).lower())
    return {w for w in words if len(w) >= 3 and w not in _NAME_STOPWORDS}


def choose_websites(p: pd.DataFrame) -> pd.DataFrame:
    """Pick one firm website per firm and fill it into every year.

    The bulk file holds only one of the URLs a firm lists in Schedule D 1.I,
    and which one changes from year to year: sometimes the firm's site,
    sometimes LinkedIn, sometimes an unrelated page (a parent company, a
    product microsite, a video-app link). Across all six years, each firm's
    candidate domains are scored:
      1. the domain contains a distinctive word from the firm name (best)
      2. otherwise, the most recently reported valid domain
    Social links are kept separately in `social_url` as a fallback."""
    p["website_candidate"] = p["website_raw"].map(clean_website)
    raw = p["website_raw"].str.strip().str.lower()
    is_social = raw.str.contains(r"linkedin|facebook|twitter|instagram|youtube|x\.com", regex=True)
    # keep only firm pages: personal LinkedIn profiles (linkedin.com/in/...) are an employee's page
    is_social &= ~raw.str.contains(r"linkedin\.com/(?:in|pub)/", regex=True, na=False)

    best = {}
    for sec, grp in p[["sec_number", "year", "firm_name", "website_candidate"]].groupby("sec_number"):
        tokens = set().union(*(_name_tokens(n) for n in grp["firm_name"].unique()))
        cands = grp.dropna(subset=["website_candidate"]).sort_values("year", ascending=False)
        if cands.empty:
            continue
        match = [w for w in cands["website_candidate"]
                 if any(t in w.replace("-", "") for t in tokens)]
        best[sec] = (match[0], "name_match") if match else (cands["website_candidate"].iloc[0], "latest")
    p["website"] = p["sec_number"].map(lambda s: best.get(s, (np.nan, None))[0])
    p["website_source"] = p["sec_number"].map(lambda s: best.get(s, (None, "none"))[1])
    p["website_filled_flag"] = p["website"].notna() & (p["website"] != p["website_candidate"])
    latest_social = raw.where(is_social).groupby(p["sec_number"]).transform("last")
    p["social_url"] = latest_social.where(p["website"].isna())
    return p.drop(columns="website_candidate")


def load_snapshot(raw_dir: Path, snap: Snapshot) -> pd.DataFrame:
    """Load + standardise one annual snapshot into clean columns."""
    df = _read_raw(raw_dir / snap.file).rename(columns=BASE_COLS)
    for col in BASE_COLS.values():          # older files lack some items (e.g. 4A)
        if col not in df:
            df[col] = ""
    for col in NUMERIC:
        df[col] = _to_number(df[col])
    for col in ["firm_name", "legal_name", "city", "state", "country", "sec_status"]:
        df[col] = df[col].str.strip()
    df["sec_number"] = df["sec_number"].str.strip().str.upper()
    # CSV years use MM/DD/YYYY; the Excel years come through as ISO timestamps
    df["filing_date"] = pd.to_datetime(df["filing_date"].str.strip(), format="mixed",
                                       errors="coerce")
    df["year"] = snap.year
    df["snapshot_date"] = pd.Timestamp(snap.as_of)
    df["source_file"] = snap.file
    return df


def build_panel(raw_dir: str | Path, cache_dir: str | Path | None = None
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (panel, qa) — the firm-year panel and a per-year data-quality log.

    If cache_dir is given, each parsed snapshot is cached there as a pickle
    (parsing the 20 MB Excel files takes ~15 s each)."""
    raw_dir = Path(raw_dir)
    frames, qa_rows = [], []
    for snap in SNAPSHOTS:
        cache = Path(cache_dir) / f"{snap.file}.pkl" if cache_dir else None
        if cache is not None and cache.exists():
            df = pd.read_pickle(cache)
        else:
            df = load_snapshot(raw_dir, snap)
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                df.to_pickle(cache)
        n_raw = len(df)

        # 1. valid SEC file numbers only (801-XXXXX); exempt reporting advisers
        #    (802-) are not in these files but we guard anyway
        valid = df["sec_number"].str.fullmatch(r"801-\d{1,6}")
        # 2. one row per firm (the SEC files should already be unique; if a
        #    firm appears twice keep its most recent filing)
        df = (df[valid].sort_values("filing_date")
              .drop_duplicates("sec_number", keep="last"))
        frames.append(df)
        qa_rows.append({
            "year": snap.year, "file": snap.file, "rows_raw": n_raw,
            "invalid_sec_number": int((~valid).sum()),
            "duplicates_removed": int(valid.sum() - len(df)),
            "firms": len(df),
            "missing_aum": int(df["aum"].isna().sum()),
            "zero_aum": int((df["aum"] == 0).sum()),
            "zero_or_missing_advisors": int((df["advisors"].fillna(0) == 0).sum()),
        })
    panel = pd.concat(frames, ignore_index=True)
    panel = derive_fields(panel)
    return panel, pd.DataFrame(qa_rows)


# --------------------------------------------------------------------------
# Derived fields
# --------------------------------------------------------------------------


def derive_fields(p: pd.DataFrame) -> pd.DataFrame:
    p = p.copy().sort_values(["sec_number", "year"])
    p = choose_websites(p)

    # ---- clients -----------------------------------------------------------
    counts = p[[f"clients_{t}" for t in CLIENT_TYPES]].fillna(0)
    # 5.D(2) is exported as the text "Fewer than 5 clients" (or "Y" in some years)
    lt5 = p[[f"lt5_{t}" for t in CLIENT_TYPES]].apply(
        lambda s: s.astype(str).str.strip().str.upper().isin(["Y", "FEWER THAN 5 CLIENTS"]))
    # categories reported only as "fewer than 5" -> impute midpoint (2)
    lt5_only = lt5.values & (counts.values == 0)
    no_raum = p["clients_no_raum_over100"].fillna(p["clients_no_raum"]).fillna(0)
    # Data-entry guard: some filers type a dollar amount into 5.C(1)
    # (e.g. a firm reporting 2.36 billion "clients" — exactly its AUM — or a
    # credit manager reporting 300 million). Treat 5.C(1) as invalid when it is
    # >= 100k AND either (a) within 0.5x-2x of the firm's AUM, or (b) more than
    # 100x the firm's RAUM client count from 5.D.
    raum_clients = counts.sum(axis=1)
    looks_like_dollars = (no_raum >= 1e5) & (
        no_raum.between(0.5 * p["aum"], 2 * p["aum"]) | (no_raum > 100 * (raum_clients + 1)))
    p["clients_5c_invalid_flag"] = looks_like_dollars
    no_raum = no_raum.mask(looks_like_dollars, 0)
    p["clients"] = counts.sum(axis=1) + 2 * lt5_only.sum(axis=1) + no_raum
    p["clients_imputed_flag"] = lt5_only.any(axis=1)

    # ---- AUM mix: how "wealth management" is this firm? --------------------
    raum = p[[f"raum_{t}" for t in CLIENT_TYPES]].fillna(0)
    indiv = raum["raum_a"] + raum["raum_b"]            # individuals + HNW
    tot = raum.sum(axis=1).replace(0, np.nan)
    p["pct_aum_individuals"] = (indiv / tot).clip(0, 1)
    p["pct_aum_hnw"] = (raum["raum_b"] / tot).clip(0, 1)
    p["pct_aum_funds"] = ((raum["raum_d"] + raum["raum_e"] + raum["raum_f"]) / tot).clip(0, 1)
    # Nevis's ICP: firms whose AUM comes mainly from individuals/HNW
    p["wealth_manager"] = p["pct_aum_individuals"] >= 0.5

    # ---- ratios (guard divide-by-zero) -------------------------------------
    p["aum_per_client"] = p["aum"] / p["clients"].where(p["clients"] > 0)
    p["aum_per_advisor"] = p["aum"] / p["advisors"].where(p["advisors"] > 0)
    p["clients_per_advisor"] = p["clients"] / p["advisors"].where(p["advisors"] > 0)
    # Digital / platform advisers (robo-advisers, managed-account platforms,
    # TAMPs) report thousands of clients per "advisor". A human-advised wealth
    # firm typically sits at 50-300. Above 1,000 we treat the firm as a
    # platform: still in the data table, excluded from capacity rankings and
    # the Nevis target list, because its "advisors" are not client-facing.
    p["digital_or_platform"] = p["clients_per_advisor"] > 1000

    p["succession_flag"] = p["succession_flag"].str.upper().eq("Y")
    p["filing_year"] = p["filing_date"].dt.year
    return p


# --------------------------------------------------------------------------
# Segments & growth
# --------------------------------------------------------------------------

SEGMENT_BINS = [0, 1e9, 10e9, 50e9, np.inf]
SEGMENT_LABELS = ["<$1B", "$1B-$10B", "$10B-$50B", ">$50B"]


def assign_segment(aum: pd.Series) -> pd.Series:
    """<$1B, $1-10B, $10-50B, >$50B. Boundaries: lower bound inclusive."""
    return pd.cut(aum, SEGMENT_BINS, labels=SEGMENT_LABELS, right=False)


def cagr(end: pd.Series, start: pd.Series, years: int) -> pd.Series:
    """Compound annual growth rate; NaN unless both values are positive."""
    ok = (start > 0) & (end > 0)
    return ((end / start) ** (1 / years) - 1).where(ok)


def growth_table(panel: pd.DataFrame, latest_year: int,
                 horizons=(1, 3, 5)) -> pd.DataFrame:
    """Wide firm table for the latest year with AUM / advisor / client growth
    over each horizon, plus inorganic-growth flags."""
    idx = panel.set_index(["sec_number", "year"])
    latest = panel[panel["year"] == latest_year].set_index("sec_number").copy()
    aum_wide = panel.pivot(index="sec_number", columns="year", values="aum")
    yoy_multiple = aum_wide / aum_wide.shift(1, axis=1)   # AUM(t) / AUM(t-1)
    adv_wide = panel.pivot(index="sec_number", columns="year", values="advisors")
    adv_multiple = adv_wide / adv_wide.shift(1, axis=1)

    for h in horizons:
        base = idx.xs(latest_year - h, level="year")
        b = base.reindex(latest.index)
        latest[f"aum_{h}y_ago"] = b["aum"]
        latest[f"advisors_{h}y_ago"] = b["advisors"]
        latest[f"clients_{h}y_ago"] = b["clients"]
        latest[f"aum_cagr_{h}y"] = cagr(latest["aum"], b["aum"], h)
        latest[f"advisor_cagr_{h}y"] = cagr(latest["advisors"], b["advisors"], h)
        latest[f"client_cagr_{h}y"] = cagr(latest["clients"], b["clients"], h)
        # under-staffing gap: AUM growth minus advisor headcount growth
        latest[f"staffing_gap_{h}y"] = latest[f"aum_cagr_{h}y"] - latest[f"advisor_cagr_{h}y"]
        # step-change flag: AUM more than tripled in a single year inside the
        # window -> usually a merger, asset transfer between affiliated
        # registrants, a fund launch or a reporting change, not organic growth
        window = [y for y in yoy_multiple.columns if latest_year - h < y <= latest_year]
        latest[f"step_change_{h}y"] = (yoy_multiple[window].max(axis=1)
                                       .reindex(latest.index) > 3)
        # headcount step-change: advisors more than halved or more than tripled
        # in a single year. Typical when a group moves its people between
        # affiliated registrants (e.g. 2,540 -> 912 at two affiliates of one
        # asset manager), which would otherwise read as "under-staffing".
        a = adv_multiple[window].reindex(latest.index)
        latest[f"advisor_step_{h}y"] = ((a.min(axis=1) < 0.5) | (a.max(axis=1) > 3)).fillna(False)

    # inorganic-growth flag: firm reported succeeding to another adviser's
    # business (Item 4.A) in any snapshot in the window (field exists 2023+)
    succ = panel.groupby("sec_number")["succession_flag"].any()
    latest["succession_any"] = succ.reindex(latest.index).fillna(False)
    latest["segment"] = assign_segment(latest["aum"])
    return latest.reset_index()
