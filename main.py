from fastapi import FastAPI, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
import pandas as pd
from collections import defaultdict
from datetime import datetime
import json
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# ---------------- COLUMN CONSTANTS ----------------

REOPEN_DATE       = "Custom field (ReOpen Date)"
REOPEN_TIME       = "Custom field (ReOpen Time Spent)"
DEVELOPER         = "Custom field (Developer)"
ASSIGNEE          = "Assignee"
ISSUE_KEY         = "Issue key"
SUMMARY           = "Summary"
STATUS            = "Status"
SPRINT            = "Sprint"
START_DATE        = "Custom field (Start date)"
END_DATE          = "Custom field (End Date)"
DEPT              = "Custom field (Department Of Created By)"
RESOLUTION        = "Custom field (Resolution)"
RESOLUTION_DETAILS= "Custom field (Resolution Details)"
RCA               = "Custom field (RCA)"
RCA_DETAILS       = "Custom field (RCA Details (Impact Analysis))"
REOPEN_RCA        = "Custom field (Reopen RCA)"
REOPEN_RCA_DETAILS= "Custom field (Reopen RCA details)"
UTC               = "Custom field (Unit Test Case)"
UTC_REVIEWED      = "Custom field (Is Unit Testcase Reviewed)"
UTC_CASES         = "Custom field (Unit Test Cases)"
ORIGINAL_ESTIMATE = "Original estimate"

# People role columns
PEOPLE_ROLE_COLS = ["Assignee", "Reporter", "Creator"]

# Status mapping
STATUS_MAP = {
    "client release":       "client_release",
    "closed":               "client_release",
    "open":                 "open",
    "development analysis": "wip",
    "awaiting response":    "wip",
    "document review":      "wip",
    "security check":       "testing",
    "testing":              "testing",
    "in progress":          "wip",
    "in review":            "wip",
    "reopened":             "reopened",
}

def map_status(raw_status):
    s = str(raw_status).strip().lower()
    for key, cat in STATUS_MAP.items():
        if key in s:
            return cat
    return s if s else "other"


# ---------------- HELPERS ----------------

def normalize_name(value):
    if pd.isna(value):
        return None
    value = str(value).strip()
    if value == "" or value.lower() == "unassigned":
        return None
    return value


def parse_log_work_columns(chunk, log_work_cols):
    result = defaultdict(set)
    for col in log_work_cols:
        if col not in chunk.columns:
            continue
        for _, row in chunk.iterrows():
            key = row.get(ISSUE_KEY)
            raw = str(row.get(col, "")).strip()
            if not raw or raw == "nan":
                continue
            parts = raw.split(";")
            if len(parts) >= 4 and parts[0] == "time-tracking":
                date_str = parts[1].strip()
                user     = parts[2].strip()
                try:
                    secs = int(parts[3].strip())
                    result[key].add((date_str, user, secs))
                except (ValueError, IndexError):
                    pass
    return {key: sum(e[2] for e in entries) for key, entries in result.items()}


def seconds_to_display(seconds):
    try:
        seconds = int(seconds)
    except (TypeError, ValueError):
        return "0h"
    if seconds == 0:
        return "0h"
    hours = seconds / 3600
    if hours >= 1:
        return f"{hours:.1f}h"
    return f"{int(seconds / 60)}m"


def load_teams():
    path = os.path.join(os.path.dirname(__file__), "teams.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f).get("teams", [])
    return []


# ---------------- ANALYSIS ENGINE ----------------

def run_csv_analysis(file):

    chunk_size = 100_000

    total_tickets    = 0
    reopened_tickets = 0

    status_counts = defaultdict(int)

    developer_stats = defaultdict(lambda: {
        "tickets":              set(),
        "status_breakdown":     defaultdict(int),
        "time_spent_secs":      0,
        "reopened":             0,
        "original_est_secs":    0,
    })

    sprint_stats = defaultdict(lambda: {
        "total": 0, "completed": 0, "reopened": 0
    })

    dev_ticket_people = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

    ticket_dept              = {}
    ticket_status            = {}
    ticket_summary           = {}
    ticket_dev               = {}
    ticket_log_secs          = {}
    ticket_resolution        = {}
    ticket_resolution_details= {}
    ticket_rca               = {}
    ticket_rca_details       = {}
    ticket_reopen_rca        = {}
    ticket_reopen_rca_details= {}
    ticket_utc               = {}
    ticket_utc_cases         = {}
    ticket_original_estimate = {}
    ticket_reopen_date       = {}
    ticket_reopen_count      = defaultdict(int)

    reopen_history    = []
    all_dates         = []
    all_sprints       = set()

    first_chunk   = True
    log_work_cols = []

    for chunk in pd.read_csv(file, chunksize=chunk_size, low_memory=False):

        total_tickets += len(chunk)

        if first_chunk:
            log_work_cols = [c for c in chunk.columns
                             if "log work" in c.lower() and "id" not in c.lower()]
            first_chunk = False

        # Collect sprint names
        if SPRINT in chunk.columns:
            for s in chunk[SPRINT].dropna():
                sv = str(s).strip()
                if sv and sv.lower() != "nan":
                    all_sprints.add(sv)

        # Reopen mask
        if REOPEN_DATE in chunk.columns:
            reopened_mask = chunk[REOPEN_DATE].notna()
            reopened_tickets += int(reopened_mask.sum())
        else:
            reopened_mask = pd.Series([False] * len(chunk), index=chunk.index)

        # Log work
        chunk_log = parse_log_work_columns(chunk, log_work_cols)
        for k, secs in chunk_log.items():
            ticket_log_secs[k] = ticket_log_secs.get(k, 0) + secs

        # Dates
        for col in [START_DATE, END_DATE, "Created", "Resolved"]:
            if col in chunk.columns:
                parsed = pd.to_datetime(chunk[col], errors="coerce")
                all_dates.extend(parsed.dropna().tolist())

        # Status counts
        if STATUS in chunk.columns:
            for s in chunk[STATUS].dropna():
                status_counts[map_status(s)] += 1

        # Dept
        if DEPT in chunk.columns:
            for _, row in chunk.iterrows():
                key  = row.get(ISSUE_KEY)
                dept = normalize_name(row.get(DEPT))
                if key and dept:
                    ticket_dept[key] = dept

        # Custom fields
        for field_const, store in [
            (RESOLUTION,          ticket_resolution),
            (RESOLUTION_DETAILS,  ticket_resolution_details),
            (RCA,                ticket_rca),
            (RCA_DETAILS,        ticket_rca_details),
            (REOPEN_RCA,         ticket_reopen_rca),
            (REOPEN_RCA_DETAILS, ticket_reopen_rca_details),
            (UTC,                ticket_utc),
            (UTC_REVIEWED,       ticket_utc),
            (UTC_CASES,          ticket_utc_cases),
            (ORIGINAL_ESTIMATE,  ticket_original_estimate),
        ]:
            if field_const in chunk.columns:
                for _, row in chunk.iterrows():
                    key = row.get(ISSUE_KEY)
                    val = row.get(field_const)
                    if key and not pd.isna(val) and str(val).strip():
                        store[key] = str(val).strip()

        # Reopen date + count per ticket
        if REOPEN_DATE in chunk.columns:
            for _, row in chunk[reopened_mask].iterrows():
                key = row.get(ISSUE_KEY)
                if key:
                    ticket_reopen_date[key] = row.get(REOPEN_DATE, "")
                    ticket_reopen_count[key] += 1

        dev_col_exists = DEVELOPER in chunk.columns

        for idx, row in chunk.iterrows():
            key        = row.get(ISSUE_KEY)
            raw_status = str(row.get(STATUS, ""))
            status_cat = map_status(raw_status)
            sprint     = str(row.get(SPRINT, "Unknown"))
            is_reopen  = reopened_mask.loc[idx]

            if key:
                ticket_status[key]  = raw_status
                ticket_summary[key] = row.get(SUMMARY, "")

            dev = None
            if dev_col_exists:
                dev = normalize_name(row.get(DEVELOPER))
            if not dev:
                dev = normalize_name(row.get(ASSIGNEE))

            if key and dev:
                ticket_dev[key] = dev

            sprint_stats[sprint]["total"] += 1
            if status_cat == "client_release":
                sprint_stats[sprint]["completed"] += 1
            if is_reopen:
                sprint_stats[sprint]["reopened"] += 1

            if dev and key:
                developer_stats[dev]["tickets"].add(key)
                developer_stats[dev]["status_breakdown"][status_cat] += 1
                if is_reopen:
                    developer_stats[dev]["reopened"] += 1

                for role_col in PEOPLE_ROLE_COLS:
                    if role_col in chunk.columns:
                        name = normalize_name(row.get(role_col))
                        if name:
                            dev_ticket_people[dev][key][role_col].add(name)

        # Reopen history
        if {ISSUE_KEY, SUMMARY}.issubset(chunk.columns):
            for _, row in chunk[reopened_mask].iterrows():
                reopen_history.append({
                    "key":           row.get(ISSUE_KEY),
                    "summary":       row.get(SUMMARY),
                    "assignee":      row.get(ASSIGNEE, ""),
                    "developer":     normalize_name(row.get(DEVELOPER)) if DEVELOPER in chunk.columns else row.get(ASSIGNEE, ""),
                    "rework_effort": row.get(REOPEN_TIME, 0) if REOPEN_TIME in chunk.columns else 0,
                    "sprint":        row.get(SPRINT, "Unknown"),
                    "start_date":    row.get(START_DATE, "") if START_DATE in chunk.columns else "",
                    "reopen_date":   row.get(REOPEN_DATE, "") if REOPEN_DATE in chunk.columns else "",
                    "status":        row.get(STATUS, ""),
                })

    # Add log work time to developer
    for key, secs in ticket_log_secs.items():
        dev = ticket_dev.get(key)
        if dev:
            developer_stats[dev]["time_spent_secs"] += secs

    # Add original estimate secs per developer
    for key, raw_est in ticket_original_estimate.items():
        dev = ticket_dev.get(key)
        if dev:
            try:
                est_secs = int(float(raw_est)) if raw_est else 0
            except (ValueError, TypeError):
                est_secs = 0
            developer_stats[dev]["original_est_secs"] += est_secs

    # ---- FINAL CALCULATIONS ----

    reopen_rate      = round((reopened_tickets / total_tickets * 100), 2) if total_tickets else 0
    sorted_dates     = sorted(all_dates)
    date_range_start = sorted_dates[-1].strftime("%Y-%m-%d") if sorted_dates else "N/A"
    date_range_end   = sorted_dates[0].strftime("%Y-%m-%d")  if sorted_dates else "N/A"

    status_counts["reopened"] = int(reopened_tickets)

    total_rework = seconds_to_display(
        sum(ticket_log_secs.get(t["key"], 0) for t in reopen_history if t.get("key"))
    )

    # Sprint label — prefer shortest meaningful name or comma-join if multiple
    sprint_label = ", ".join(sorted(all_sprints)) if all_sprints else "Unknown"

    # Developer summary
    developer_summary = []
    for dev, stats in developer_stats.items():
        dev_tickets = stats["tickets"]
        a           = len(dev_tickets)
        r           = stats["reopened"]
        secs        = stats["time_spent_secs"]
        est_secs    = stats["original_est_secs"]
        breakdown   = dict(stats["status_breakdown"])
        c           = breakdown.get("client_release", 0)

        completion_pct = round((c / a * 100), 1) if a else 0
        reopen_pct     = round((r / a * 100), 1) if a else 0

        ticket_detail = []
        for ticket_key in sorted(dev_tickets):
            roles_for_ticket = {
                role: sorted(list(names))
                for role, names in dev_ticket_people[dev][ticket_key].items()
            }

            raw_est = ticket_original_estimate.get(ticket_key, 0)
            try:
                est_secs_t = int(float(raw_est)) if raw_est else 0
            except (ValueError, TypeError):
                est_secs_t = 0

            rca_data = {
                "rca":                ticket_rca.get(ticket_key, ""),
                "rca_details":        ticket_rca_details.get(ticket_key, ""),
                "reopen_rca":         ticket_reopen_rca.get(ticket_key, ""),
                "reopen_rca_details": ticket_reopen_rca_details.get(ticket_key, ""),
            }
            has_rca = any(v for v in rca_data.values())

            t_status  = ticket_status.get(ticket_key, "")
            res_val   = ticket_resolution.get(ticket_key, "")
            res_det   = ticket_resolution_details.get(ticket_key, "")
            utc_val   = ticket_utc.get(ticket_key, "")
            utc_cases = ticket_utc_cases.get(ticket_key, "")

            ticket_detail.append({
                "key":                ticket_key,
                "summary":            ticket_summary.get(ticket_key, ""),
                "status":             t_status,
                "status_cat":         map_status(t_status),
                "time":               seconds_to_display(ticket_log_secs.get(ticket_key, 0)),
                "original_estimate":  seconds_to_display(est_secs_t),
                "resolution":         res_val,
                "resolution_details": res_det,
                "has_resolution":     bool((res_val and res_val.strip()) or (res_det and res_det.strip())),
                "rca_data":           rca_data,
                "has_rca":            has_rca,
                "utc":                utc_val,
                "utc_cases":          utc_cases,
                "has_utc":            bool((utc_val and utc_val.strip()) or (utc_cases and utc_cases.strip())),
                "people":             roles_for_ticket,
                "is_reopened":        ticket_key in ticket_reopen_count,
                "is_cr":              map_status(t_status) == "client_release",
            })

        developer_summary.append({
            "name":              dev,
            "assigned":          a,
            "completed":         c,
            "reopened":          r,
            "completion_pct":    completion_pct,
            "reopen_pct":        reopen_pct,
            "time_spent":        seconds_to_display(secs),
            "time_spent_secs":   secs,
            "total_original_est": seconds_to_display(est_secs),
            "status_breakdown":  breakdown,
            "ticket_detail":     ticket_detail,
        })

    # Sprint details (kept for internal use but not surfaced as table)
    sprint_details = []
    for sprint_name, stats in sprint_stats.items():
        t = stats["total"]
        r = stats["reopened"]
        sprint_details.append({
            "name":        sprint_name,
            "total":       t,
            "completed":   stats["completed"],
            "reopened":    r,
            "reopen_rate": round((r / t * 100), 1) if t else 0,
        })

    # Reopen analysis — per ticket with reopen info
    reopen_analysis = []
    seen_keys = set()
    for item in reopen_history:
        key = item.get("key")
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        dev = ticket_dev.get(key, item.get("assignee", ""))
        reopen_analysis.append({
            "key":          key,
            "summary":      ticket_summary.get(key, item.get("summary", "")),
            "developer":    dev,
            "status":       ticket_status.get(key, item.get("status", "")),
            "reopen_date":  ticket_reopen_date.get(key, item.get("reopen_date", "")),
            "start_date":   item.get("start_date", ""),
            "reopen_count": ticket_reopen_count.get(key, 1),
            "sprint":       item.get("sprint", ""),
        })

    # Team summary
    teams_raw    = load_teams()
    team_summary = []
    for team in teams_raw:
        lead       = team["lead"]
        members    = team["members"]
        team_entry = {"lead": lead, "members": []}

        for person in [lead] + members:
            if person in developer_stats:
                stats     = developer_stats[person]
                secs      = stats["time_spent_secs"]
                breakdown = dict(stats["status_breakdown"])
                assigned  = len(stats["tickets"])
                reopened  = stats["reopened"]
            else:
                secs = 0; breakdown = {}; assigned = 0; reopened = 0

            team_entry["members"].append({
                "name":             person,
                "is_lead":          person == lead,
                "assigned":         assigned,
                "status_breakdown": breakdown,
                "time_spent":       seconds_to_display(secs),
                "reopened":         reopened,
                "reopen_rate":      round((reopened / assigned * 100), 1) if assigned else 0,
            })

        team_summary.append(team_entry)

    return {
        "total":             total_tickets,
        "reopened":          int(reopened_tickets),
        "rate":              reopen_rate,
        "total_rework":      total_rework,
        "sprint_label":      sprint_label,
        "status_counts":     dict(status_counts),
        "developer_stats":   developer_summary,
        "sprint_details":    sprint_details,
        "teams":             team_summary,
        "reopen_analysis":   reopen_analysis,
        "date_range_start":  date_range_start,
        "date_range_end":    date_range_end,
        "generated_at":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "reopen_history":    reopen_history,
    }


# ---------------- ROUTES ----------------

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analysis")
async def analysis(request: Request, file: UploadFile = File(...)):
    data = run_csv_analysis(file.file)
    return templates.TemplateResponse(
        "dashboard.html", {"request": request, "data": data}
    )
