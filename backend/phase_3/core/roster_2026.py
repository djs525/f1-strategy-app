"""
phase_3/core/roster_2026.py
=============================
The authoritative 22-driver 2026 grid.
Used by the championship tracker, insights engine, and API routes.
"""

# Full driver code list
ROSTER_2026 = [
    "NOR", "PIA",   # McLaren
    "LEC", "HAM",   # Ferrari
    "VER", "HAD",   # Red Bull
    "RUS", "ANT",   # Mercedes
    "ALO", "STR",   # Aston Martin
    "GAS", "COL",   # Alpine
    "OCO", "BEA",   # Haas
    "LAW", "LIN",   # Racing Bulls
    "ALB", "SAI",   # Williams
    "HUL", "BOR",   # Audi
    "PER", "BOT",   # Cadillac
]

# Team → driver mapping
TEAMS_2026 = {
    "McLaren":           ["NOR", "PIA"],
    "Ferrari":           ["LEC", "HAM"],
    "Red Bull":          ["VER", "HAD"],
    "Mercedes":          ["RUS", "ANT"],
    "Aston Martin":      ["ALO", "STR"],
    "Alpine":            ["GAS", "COL"],
    "Haas":              ["OCO", "BEA"],
    "Racing Bulls":      ["LAW", "LIN"],
    "Williams":          ["ALB", "SAI"],
    "Audi":              ["HUL", "BOR"],
    "Cadillac F1 Team":  ["PER", "BOT"],
}

# 2026 F1 Race Calendar
CALENDAR_2026 = [
    {"round": 1, "country": "Australia", "circuit": "Albert Park", "date": "2026-03-08"},
    {"round": 2, "country": "China", "circuit": "Shanghai", "date": "2026-03-15"},
    {"round": 3, "country": "Japan", "circuit": "Suzuka", "date": "2026-03-29"},
    {"round": 4, "country": "Bahrain", "circuit": "Sakhir", "date": "2026-04-12"},
    {"round": 5, "country": "Saudi Arabia", "circuit": "Jeddah", "date": "2026-04-19"},
    {"round": 6, "country": "United States", "circuit": "Miami", "date": "2026-05-03"},
    {"round": 7, "country": "Canada", "circuit": "Montreal", "date": "2026-05-24"},
    {"round": 8, "country": "Monaco", "circuit": "Monaco", "date": "2026-06-07"},
    {"round": 9, "country": "Spain", "circuit": "Barcelona", "date": "2026-06-14"},
    {"round": 10, "country": "Austria", "circuit": "Red Bull Ring", "date": "2026-06-28"},
    {"round": 11, "country": "Great Britain", "circuit": "Silverstone", "date": "2026-07-05"},
    {"round": 12, "country": "Belgium", "circuit": "Spa-Francorchamps", "date": "2026-07-19"},
    {"round": 13, "country": "Hungary", "circuit": "Hungaroring", "date": "2026-07-26"},
    {"round": 14, "country": "Netherlands", "circuit": "Zandvoort", "date": "2026-08-23"},
    {"round": 15, "country": "Italy", "circuit": "Monza", "date": "2026-09-06"},
    {"round": 16, "country": "Spain", "circuit": "Madrid", "date": "2026-09-13"},
    {"round": 17, "country": "Azerbaijan", "circuit": "Baku", "date": "2026-09-26"},
    {"round": 18, "country": "Singapore", "circuit": "Marina Bay", "date": "2026-10-11"},
    {"round": 19, "country": "United States", "circuit": "Austin", "date": "2026-10-25"},
    {"round": 20, "country": "Mexico", "circuit": "Mexico City", "date": "2026-11-01"},
    {"round": 21, "country": "Brazil", "circuit": "Interlagos", "date": "2026-11-08"},
    {"round": 22, "country": "United States", "circuit": "Las Vegas", "date": "2026-11-21"},
    {"round": 23, "country": "Qatar", "circuit": "Lusail", "date": "2026-11-29"},
    {"round": 24, "country": "Abu Dhabi", "circuit": "Yas Marina", "date": "2026-12-06"}
]

# Track type classifications
CIRCUIT_TYPES = {
    "Albert Park": "Semi-Permanent",
    "Shanghai": "Permanent",
    "Suzuka": "Permanent",
    "Sakhir": "Permanent",
    "Jeddah": "Street",
    "Miami": "Street",
    "Montreal": "Semi-Permanent",
    "Monaco": "Street",
    "Barcelona": "Permanent",
    "Red Bull Ring": "Permanent",
    "Silverstone": "Permanent",
    "Spa-Francorchamps": "Permanent",
    "Hungaroring": "Permanent",
    "Zandvoort": "Permanent",
    "Monza": "Permanent",
    "Madrid": "Street",
    "Baku": "Street",
    "Marina Bay": "Street",
    "Austin": "Permanent",
    "Mexico City": "Permanent",
    "Interlagos": "Permanent",
    "Las Vegas": "Street",
    "Lusail": "Permanent",
    "Yas Marina": "Permanent"
}
# Car numbers for display
CAR_NUMBERS = {
    "NOR": 1,  "PIA": 81,
    "LEC": 16, "HAM": 44,
    "VER": 3,  "HAD": 6,
    "RUS": 63, "ANT": 12,
    "ALO": 14, "STR": 18,
    "GAS": 10, "COL": 43,
    "OCO": 31, "BEA": 87,
    "LAW": 30, "LIN": 41,
    "ALB": 23, "SAI": 55,
    "HUL": 27, "BOR": 5,
    "PER": 11, "BOT": 77,
}