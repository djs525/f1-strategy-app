"""
drivers_config.py
=================
The single source of truth for all 21 drivers we are scraping
(Arvid Lindblad excluded — insufficient historical F1 data).

THEORY — Why a config file instead of hardcoding in the scraper?
-----------------------------------------------------------------
This is the "separation of concerns" principle. Data (who to scrape)
lives here. Logic (how to scrape) lives in scrape_driver_history.py.
This means if a driver retires or a new one joins, you change ONE line
here and the entire pipeline updates automatically. No hunting through
thousands of lines of scraping code.

Jolpica Driver ID slugs:
    These are the URL-safe identifiers used by the Jolpica-F1 API
    (the community-maintained successor to the now-deprecated Ergast API).
    Example endpoint:
        https://api.jolpi.ca/ergast/f1/2021/drivers/hamilton/results.json
    The slug is always lowercase, using the driver's surname.
    Exceptions are noted inline where slugs are non-obvious.
"""

# =============================================================================
# DRIVER REGISTRY
# =============================================================================

DRIVERS = {

    # -------------------------------------------------------------------------
    # VETERANS — 10+ seasons of data.
    # -------------------------------------------------------------------------

    "HAM": {
        "jolpica_id": "hamilton",
        "name": "Lewis Hamilton",
        "debut_year": 2007,
        "active_years": list(range(2007, 2026)),
        "team_2026": "Ferrari",
        "is_rookie_2026": False,
        "notes": "19 seasons, 7 WDC titles. 2007-2012 McLaren, 2013-2024 Mercedes, 2025 Ferrari.",
    },

    "ALO": {
        "jolpica_id": "alonso",
        "name": "Fernando Alonso",
        "debut_year": 2001,
        "active_years": list(range(2001, 2019)) + list(range(2021, 2026)),
        "team_2026": "Aston Martin",
        "is_rookie_2026": False,
        "notes": "Oldest active driver. Note the 2019-2020 gap.",
    },

    "BOT": {
        "jolpica_id": "bottas",
        "name": "Valtteri Bottas",
        "debut_year": 2013,
        "active_years": list(range(2013, 2025)),
        "team_2026": "Cadillac F1 Team",
        "is_rookie_2026": False,
        "notes": "Sat out 2025 F1 season. Cleanest 'same-car, different-driver' dataset vs Hamilton.",
    },

    "PER": {
        "jolpica_id": "perez",
        "name": "Sergio Perez",
        "debut_year": 2011,
        "active_years": list(range(2011, 2025)),
        "team_2026": "Cadillac F1 Team",
        "is_rookie_2026": False,
        "notes": "Historically elite on street circuits. Dropped by Red Bull end of 2024.",
    },

    "HUL": {
        "jolpica_id": "hulkenberg",
        "name": "Nico Hulkenberg",
        "debut_year": 2010,
        "active_years": list(range(2010, 2020)) + [2020] + list(range(2023, 2026)),
        "team_2026": "Audi",
        "is_rookie_2026": False,
        "notes": "Sat out 2021-2022 as reserve driver. 2020 counted as 3 sub appearances.",
    },

    "STR": {
        "jolpica_id": "stroll",
        "name": "Lance Stroll",
        "debut_year": 2017,
        "active_years": list(range(2017, 2026)),
        "team_2026": "Aston Martin",
        "is_rookie_2026": False,
        "notes": "Continuous tenure at effectively the same outfit under a rebrand.",
    },

    # -------------------------------------------------------------------------
    # MID-CAREER DRIVERS — 7-11 seasons.
    # -------------------------------------------------------------------------

    "SAI": {
        "jolpica_id": "sainz",
        "name": "Carlos Sainz",
        "debut_year": 2015,
        "active_years": list(range(2015, 2026)),
        "team_2026": "Williams",
        "is_rookie_2026": False,
        "notes": "Debuted mid-2015, so race count for that year is partial.",
    },

    "GAS": {
        "jolpica_id": "gasly",
        "name": "Pierre Gasly",
        "debut_year": 2017,
        "active_years": list(range(2017, 2026)),
        "team_2026": "Alpine",
        "is_rookie_2026": False,
        "notes": "Model learns how demotion affects performance psychology (Red Bull 2019).",
    },

    "VER": {
        "jolpica_id": "max_verstappen",
        "name": "Max Verstappen",
        "car_number": 33,
        "debut_year": 2015,
        "active_years": list(range(2015, 2026)),
        "team_2026": "Red Bull",
        "is_rookie_2026": False,
        "notes": "4x WDC. Reverts to #33 when not reigning champion.",
    },

    "LEC": {
        "jolpica_id": "leclerc",
        "name": "Charles Leclerc",
        "debut_year": 2018,
        "active_years": list(range(2018, 2026)),
        "team_2026": "Ferrari",
        "is_rookie_2026": False,
        "notes": "Fastest testing time in 2026 Bahrain testing.",
    },

    "NOR": {
        "jolpica_id": "norris",
        "name": "Lando Norris",
        "car_number": 1,
        "debut_year": 2019,
        "active_years": list(range(2019, 2026)),
        "team_2026": "McLaren",
        "is_rookie_2026": False,
        "notes": "Reigning 2025 World Drivers Champion. Earns the #1.",
    },

    "RUS": {
        "jolpica_id": "russell",
        "name": "George Russell",
        "debut_year": 2019,
        "active_years": list(range(2019, 2026)),
        "team_2026": "Mercedes",
        "is_rookie_2026": False,
        "notes": "His Williams years are some of the purest driver-skill datapoints.",
    },

    "ALB": {
        "jolpica_id": "albon",
        "name": "Alexander Albon",
        "debut_year": 2019,
        "active_years": [2019, 2020] + list(range(2022, 2026)),
        "team_2026": "Williams",
        "is_rookie_2026": False,
        "notes": "2021 gap year (no F1 race starts).",
    },

    "OCO": {
        "jolpica_id": "ocon",
        "name": "Esteban Ocon",
        "debut_year": 2016,
        "active_years": [2016, 2017, 2018] + list(range(2020, 2026)),
        "team_2026": "Haas",
        "is_rookie_2026": False,
        "notes": "2019 gap (Mercedes reserve).",
    },

    # -------------------------------------------------------------------------
    # NEWER DRIVERS — 1-3 seasons.
    # -------------------------------------------------------------------------

    "PIA": {
        "jolpica_id": "piastri",
        "name": "Oscar Piastri",
        "debut_year": 2023,
        "active_years": list(range(2023, 2026)),
        "team_2026": "McLaren",
        "is_rookie_2026": False,
        "notes": "Sat out 2022 as Alpine reserve.",
    },

    "LAW": {
        "jolpica_id": "lawson",
        "name": "Liam Lawson",
        "debut_year": 2023,
        "active_years": list(range(2023, 2026)),
        "team_2026": "Racing Bulls",
        "is_rookie_2026": False,
        "notes": "Partial seasons flagged with debut_race_flag.",
    },

    "BEA": {
        "jolpica_id": "bearman",
        "name": "Oliver Bearman",
        "debut_year": 2024,
        "active_years": [2024, 2025],
        "team_2026": "Haas",
        "is_rookie_2026": False,
        "notes": "Debuted as sub at Ferrari 2024. Very thin data.",
    },

    "COL": {
        "jolpica_id": "colapinto",
        "name": "Franco Colapinto",
        "debut_year": 2024,
        "active_years": [2024, 2025],
        "team_2026": "Alpine",
        "is_rookie_2026": False,
        "notes": "Debuted mid-2024 at Williams.",
    },

    "ANT": {
        "jolpica_id": "antonelli",
        "name": "Kimi Antonelli",
        "debut_year": 2025,
        "active_years": [2025],
        "team_2026": "Mercedes",
        "is_rookie_2026": False,
        "notes": "Mercedes junior, F3 champion. Relies on population priors.",
    },

    "BOR": {
        "jolpica_id": "bortoleto",
        "name": "Gabriel Bortoleto",
        "debut_year": 2025,
        "active_years": [2025],
        "team_2026": "Audi",
        "is_rookie_2026": False,
        "notes": "F2 Champion 2024.",
    },

    "HAD": {
        "jolpica_id": "hadjar",
        "name": "Isack Hadjar",
        "debut_year": 2025,
        "active_years": [2025],
        "team_2026": "Red Bull",
        "is_rookie_2026": False,
        "notes": "F2 Championship runner-up 2024. Steps up to Red Bull.",
    },
}

# =============================================================================
# CONVENIENCE EXPORTS
# =============================================================================

ALL_DRIVER_IDS = list(DRIVERS.keys())

TIER_VETERAN   = [k for k, v in DRIVERS.items() if v["debut_year"] <= 2015]
TIER_MID       = [k for k, v in DRIVERS.items() if 2016 <= v["debut_year"] <= 2021]
TIER_NEWCOMER  = [k for k, v in DRIVERS.items() if v["debut_year"] >= 2022]

# Ensure Cadillac F1 Team exactly matches roster_2026.py!
TEAMS_2026 = {
    "McLaren":           ["NOR","PIA"],
    "Ferrari":           ["LEC","HAM"],
    "Red Bull":          ["VER", "HAD"],
    "Mercedes":          ["RUS", "ANT"],
    "Aston Martin":      ["ALO","STR"],
    "Alpine":            ["GAS","COL"],
    "Haas":              ["OCO","BEA"],
    "Racing Bulls":      ["LAW","LIN"],
    "Williams":          ["ALB","SAI"],
    "Audi":              ["HUL", "BOR"],
    "Cadillac F1 Team":  ["PER", "BOT"],
}

CALENDAR_2026 = [
    {"round": 1,  "name": "Australian Grand Prix",     "circuit": "albert_park",    "sprint": False},
    {"round": 2,  "name": "Chinese Grand Prix",        "circuit": "shanghai",        "sprint": True},
    {"round": 3,  "name": "Japanese Grand Prix",       "circuit": "suzuka",          "sprint": False},
    {"round": 4,  "name": "Bahrain Grand Prix",        "circuit": "bahrain",         "sprint": False},
    {"round": 5,  "name": "Saudi Arabian Grand Prix",  "circuit": "jeddah",          "sprint": False},
    {"round": 6,  "name": "Miami Grand Prix",          "circuit": "miami",           "sprint": True},
    {"round": 7,  "name": "Canadian Grand Prix",       "circuit": "villeneuve",      "sprint": True},
    {"round": 8,  "name": "Monaco Grand Prix",         "circuit": "monaco",          "sprint": False},
    {"round": 9,  "name": "Spanish Grand Prix",        "circuit": "catalunya",       "sprint": False},
    {"round": 10, "name": "Austrian Grand Prix",       "circuit": "red_bull_ring",   "sprint": False},
    {"round": 11, "name": "British Grand Prix",        "circuit": "silverstone",     "sprint": True},
    {"round": 12, "name": "Belgian Grand Prix",        "circuit": "spa",             "sprint": False},
    {"round": 13, "name": "Hungarian Grand Prix",      "circuit": "hungaroring",     "sprint": False},
    {"round": 14, "name": "Dutch Grand Prix",          "circuit": "zandvoort",       "sprint": True},
    {"round": 15, "name": "Italian Grand Prix",        "circuit": "monza",           "sprint": False},
    {"round": 16, "name": "Madrid Grand Prix",         "circuit": "madrid",          "sprint": False},
    {"round": 17, "name": "Azerbaijan Grand Prix",     "circuit": "baku",            "sprint": False},
    {"round": 18, "name": "Singapore Grand Prix",      "circuit": "marina_bay",      "sprint": True},
    {"round": 19, "name": "United States Grand Prix",  "circuit": "americas",        "sprint": False},
    {"round": 20, "name": "Mexico City Grand Prix",    "circuit": "rodriguez",       "sprint": False},
    {"round": 21, "name": "São Paulo Grand Prix",      "circuit": "interlagos",      "sprint": False},
    {"round": 22, "name": "Las Vegas Grand Prix",      "circuit": "las_vegas",       "sprint": False},
    {"round": 23, "name": "Qatar Grand Prix",          "circuit": "losail",          "sprint": False},
    {"round": 24, "name": "Abu Dhabi Grand Prix",      "circuit": "yas_marina",      "sprint": False},
]

CIRCUIT_TYPES = {
    "albert_park":  "street_hybrid",
    "shanghai":     "permanent",
    "suzuka":       "permanent_highspeed",
    "bahrain":      "permanent",
    "jeddah":       "street",
    "miami":        "street_hybrid",
    "villeneuve":   "street_hybrid",
    "monaco":       "street",
    "catalunya":    "permanent",
    "red_bull_ring":"permanent_highspeed",
    "silverstone":  "permanent_highspeed",
    "spa":          "permanent_highspeed",
    "hungaroring":  "permanent_technical",
    "zandvoort":    "permanent",
    "monza":        "permanent_highspeed",
    "madrid":       "street_hybrid",
    "baku":         "street",
    "marina_bay":   "street",
    "americas":     "permanent",
    "rodriguez":    "permanent",
    "interlagos":   "permanent",
    "las_vegas":    "street",
    "losail":       "permanent",
    "yas_marina":   "permanent",
}