import { useState, useEffect, useCallback } from "react";

// ─── CONSTANTS ────────────────────────────────────────────────────────────────
const API_BASE = "/api";

const TEAM_COLORS = {
  "McLaren":          "#FF8000",
  "Ferrari":          "#E8002D",
  "Red Bull":         "#3671C6",
  "Mercedes":         "#27F4D2",
  "Aston Martin":     "#229971",
  "Alpine":           "#FF87BC",
  "Haas":             "#B6BABD",
  "Racing Bulls":     "#6692FF",
  "Williams":         "#64C4FF",
  "Audi":             "#C8D44B",
  "Cadillac F1 Team": "#B01C38",
};

const COMPOUND_COLORS = {
  SOFT:         "#E8002D",
  MEDIUM:       "#FFF200",
  HARD:         "#FFFFFF",
  INTERMEDIATE: "#39B54A",
  WET:          "#0067FF",
};

const DRIVERS_2026 = [
  { code: "NOR", name: "Lando Norris",      team: "McLaren",          number: 1  },
  { code: "PIA", name: "Oscar Piastri",     team: "McLaren",          number: 81 },
  { code: "LEC", name: "Charles Leclerc",   team: "Ferrari",          number: 16 },
  { code: "HAM", name: "Lewis Hamilton",    team: "Ferrari",          number: 44 },
  { code: "VER", name: "Max Verstappen",    team: "Red Bull",         number: 3  },
  { code: "HAD", name: "Isack Hadjar",      team: "Red Bull",         number: 6  },
  { code: "RUS", name: "George Russell",    team: "Mercedes",         number: 63 },
  { code: "ANT", name: "Kimi Antonelli",    team: "Mercedes",         number: 12 },
  { code: "ALO", name: "Fernando Alonso",   team: "Aston Martin",     number: 14 },
  { code: "STR", name: "Lance Stroll",      team: "Aston Martin",     number: 18 },
  { code: "GAS", name: "Pierre Gasly",      team: "Alpine",           number: 10 },
  { code: "COL", name: "Franco Colapinto",  team: "Alpine",           number: 43 },
  { code: "OCO", name: "Esteban Ocon",      team: "Haas",             number: 31 },
  { code: "BEA", name: "Oliver Bearman",    team: "Haas",             number: 87 },
  { code: "LAW", name: "Liam Lawson",       team: "Racing Bulls",     number: 30 },
  { code: "LIN", name: "Arvid Lindblad",    team: "Racing Bulls",     number: 41 },
  { code: "ALB", name: "Alexander Albon",   team: "Williams",         number: 23 },
  { code: "SAI", name: "Carlos Sainz",      team: "Williams",         number: 55 },
  { code: "HUL", name: "Nico Hulkenberg",   team: "Audi",             number: 27 },
  { code: "BOR", name: "Gabriel Bortoleto", team: "Audi",             number: 5  },
  { code: "PER", name: "Sergio Perez",      team: "Cadillac F1 Team", number: 11 },
  { code: "BOT", name: "Valtteri Bottas",   team: "Cadillac F1 Team", number: 77 },
];

const CIRCUITS_2026 = [
  { name: "Australian Grand Prix",   circuit: "Albert Park",       round: 1  },
  { name: "Chinese Grand Prix",      circuit: "Shanghai",           round: 2  },
  { name: "Japanese Grand Prix",     circuit: "Suzuka",             round: 3  },
  { name: "Bahrain Grand Prix",      circuit: "Sakhir",             round: 4  },
  { name: "Saudi Arabian Grand Prix",circuit: "Jeddah",             round: 5  },
  { name: "Miami Grand Prix",        circuit: "Miami",              round: 6  },
  { name: "Canadian Grand Prix",     circuit: "Montreal",           round: 7  },
  { name: "Monaco Grand Prix",       circuit: "Monaco",             round: 8  },
  { name: "Spanish Grand Prix",      circuit: "Barcelona",          round: 9  },
  { name: "Austrian Grand Prix",     circuit: "Red Bull Ring",      round: 10 },
  { name: "British Grand Prix",      circuit: "Silverstone",        round: 11 },
  { name: "Belgian Grand Prix",      circuit: "Spa-Francorchamps",  round: 12 },
  { name: "Hungarian Grand Prix",    circuit: "Hungaroring",        round: 13 },
  { name: "Dutch Grand Prix",        circuit: "Zandvoort",          round: 14 },
  { name: "Italian Grand Prix",      circuit: "Monza",              round: 15 },
  { name: "Madrid Grand Prix",       circuit: "Madrid",             round: 16 },
  { name: "Azerbaijan Grand Prix",   circuit: "Baku",               round: 17 },
  { name: "Singapore Grand Prix",    circuit: "Marina Bay",         round: 18 },
  { name: "United States Grand Prix",circuit: "Austin",             round: 19 },
  { name: "Mexico City Grand Prix",  circuit: "Mexico City",        round: 20 },
  { name: "São Paulo Grand Prix",    circuit: "Interlagos",         round: 21 },
  { name: "Las Vegas Grand Prix",    circuit: "Las Vegas",          round: 22 },
  { name: "Qatar Grand Prix",        circuit: "Lusail",             round: 23 },
  { name: "Abu Dhabi Grand Prix",    circuit: "Yas Marina",         round: 24 },
];

const COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"];

// Authoritative lap counts from the dataset (modal value per circuit)
const CIRCUIT_LAPS = {
  "Australian Grand Prix":      58,
  "Chinese Grand Prix":         56,
  "Japanese Grand Prix":        53,
  "Bahrain Grand Prix":         57,
  "Saudi Arabian Grand Prix":   50,
  "Miami Grand Prix":           57,
  "Canadian Grand Prix":        70,
  "Monaco Grand Prix":          78,
  "Spanish Grand Prix":         66,
  "Austrian Grand Prix":        70,
  "British Grand Prix":         52,
  "Belgian Grand Prix":         44,
  "Hungarian Grand Prix":       70,
  "Dutch Grand Prix":           72,
  "Italian Grand Prix":         53,
  "Madrid Grand Prix":          58, // analogue: Australian GP
  "Azerbaijan Grand Prix":      51,
  "Singapore Grand Prix":       62,
  "United States Grand Prix":   56,
  "Mexico City Grand Prix":     71,
  "São Paulo Grand Prix":       71,
  "Las Vegas Grand Prix":       50,
  "Qatar Grand Prix":           57,
  "Abu Dhabi Grand Prix":       58,
  // Historical-only circuits (Strategy Simulator)
  "Emilia Romagna Grand Prix":  63,
  "French Grand Prix":          53,
  "Portuguese Grand Prix":      66,
  "Turkish Grand Prix":         58,
  "Russian Grand Prix":         53,
  "Styrian Grand Prix":         70,
};

// Circuits present in the training dataset — used by Strategy Simulator
const HISTORICAL_CIRCUITS = [
  "Australian Grand Prix",
  "Bahrain Grand Prix",
  "Chinese Grand Prix",
  "Japanese Grand Prix",
  "Saudi Arabian Grand Prix",
  "Miami Grand Prix",
  "Canadian Grand Prix",
  "Monaco Grand Prix",
  "Spanish Grand Prix",
  "Austrian Grand Prix",
  "British Grand Prix",
  "Belgian Grand Prix",
  "Hungarian Grand Prix",
  "Dutch Grand Prix",
  "Italian Grand Prix",
  "Azerbaijan Grand Prix",
  "Singapore Grand Prix",
  "United States Grand Prix",
  "Mexico City Grand Prix",
  "São Paulo Grand Prix",
  "Las Vegas Grand Prix",
  "Qatar Grand Prix",
  "Abu Dhabi Grand Prix",
  "Emilia Romagna Grand Prix",
  "French Grand Prix",
  "Portuguese Grand Prix",
  "Turkish Grand Prix",
  "Russian Grand Prix",
  "Styrian Grand Prix",
];

// ─── STYLES ───────────────────────────────────────────────────────────────────
const css = `
  @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&family=Rajdhani:wght@300;400;500;600;700&family=Share+Tech+Mono&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  html, body, #root { width: 100%; min-height: 100vh; }

  :root {
    --red:      #E8002D;
    --red-dim:  #9B0020;
    --gold:     #FFD700;
    --bg:       #080A0C;
    --bg2:      #0E1216;
    --bg3:      #141A20;
    --bg4:      #1C242C;
    --border:   #1E2A35;
    --border2:  #2A3A48;
    --text:     #E8EDF2;
    --text2:    #8A9BB0;
    --text3:    #5A6B7A;
    --green:    #00C851;
    --accent:   #00A8E8;
  }

  html { font-size: 14px; }
  body { background: var(--bg); color: var(--text); font-family: 'Rajdhani', sans-serif; min-height: 100vh; overflow-x: hidden; }

  /* Scanline overlay */
  body::before {
    content: '';
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.03) 2px, rgba(0,0,0,0.03) 4px);
  }

  /* ── HEADER ── */
  .header {
    position: sticky; top: 0; z-index: 100;
    background: rgba(8,10,12,0.95);
    border-bottom: 1px solid var(--border2);
    backdrop-filter: blur(12px);
    padding: 0 2rem;
    display: flex; align-items: center; justify-content: space-between;
    height: 60px;
  }
  .header-logo {
    display: flex; align-items: center; gap: 12px;
    font-family: 'Orbitron', monospace; font-weight: 900; font-size: 1.1rem;
    letter-spacing: 0.1em; color: var(--text);
  }
  .header-logo span { color: var(--red); }
  .header-logo-mark {
    width: 32px; height: 32px;
    background: var(--red);
    clip-path: polygon(0 0, 100% 0, 80% 100%, 0 100%);
    display: flex; align-items: center; justify-content: center;
  }
  .nav-tabs {
    display: flex; gap: 2px;
  }
  .nav-tab {
    padding: 0.5rem 1.2rem;
    font-family: 'Rajdhani', sans-serif; font-weight: 600; font-size: 0.85rem;
    letter-spacing: 0.08em; text-transform: uppercase;
    background: transparent; border: none; cursor: pointer;
    color: var(--text3); transition: all 0.2s;
    position: relative;
  }
  .nav-tab::after {
    content: ''; position: absolute; bottom: -1px; left: 0; right: 0; height: 2px;
    background: var(--red); transform: scaleX(0); transition: transform 0.2s;
  }
  .nav-tab:hover { color: var(--text2); }
  .nav-tab.active { color: var(--text); }
  .nav-tab.active::after { transform: scaleX(1); }
  .api-indicator {
    display: flex; align-items: center; gap: 6px;
    font-family: 'Share Tech Mono', monospace; font-size: 0.72rem; color: var(--text3);
  }
  .api-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--text3);
  }
  .api-dot.online { background: var(--green); box-shadow: 0 0 8px var(--green); animation: pulse 2s infinite; }
  .api-dot.offline { background: var(--red); }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

  /* ── LAYOUT ── */
  .main { width: 100%; padding: 1.5rem 2rem; position: relative; z-index: 1; box-sizing: border-box; }

  /* ── SECTION HEADING ── */
  .section-head { margin-bottom: 2rem; }
  .section-title {
    font-family: 'Orbitron', monospace; font-weight: 700; font-size: 1.4rem;
    letter-spacing: 0.06em; color: var(--text);
    display: flex; align-items: center; gap: 1rem;
  }
  .section-title::before {
    content: ''; display: block; width: 4px; height: 1.6rem;
    background: var(--red);
  }
  .section-sub { color: var(--text3); font-size: 0.9rem; margin-top: 0.4rem; padding-left: 1.2rem; letter-spacing: 0.04em; }

  /* ── PANELS / CARDS ── */
  .panel {
    background: var(--bg2); border: 1px solid var(--border);
    border-radius: 2px; position: relative; overflow: hidden;
  }
  .panel::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, var(--red), transparent);
  }
  .panel-header {
    padding: 1rem 1.5rem; border-bottom: 1px solid var(--border);
    font-family: 'Orbitron', monospace; font-size: 0.75rem; font-weight: 500;
    letter-spacing: 0.12em; text-transform: uppercase; color: var(--text2);
    display: flex; align-items: center; gap: 8px;
  }
  .panel-body { padding: 1.5rem; }

  /* ── GRID LAYOUTS ── */
  .grid-2 { display: grid; grid-template-columns: minmax(0,1fr) minmax(0,1fr); gap: 1.5rem; }
  .grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; }
  .grid-auto { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 0.75rem; }

  /* ── FORM ELEMENTS ── */
  .field { display: flex; flex-direction: column; gap: 6px; }
  .field-label {
    font-family: 'Share Tech Mono', monospace; font-size: 0.72rem;
    letter-spacing: 0.1em; text-transform: uppercase; color: var(--text3);
  }
  .field-input, .field-select {
    background: var(--bg3); border: 1px solid var(--border2);
    color: var(--text); font-family: 'Rajdhani', sans-serif; font-size: 0.95rem; font-weight: 500;
    padding: 0.6rem 0.9rem; border-radius: 2px; outline: none;
    transition: border-color 0.2s;
    width: 100%;
  }
  .field-input:focus, .field-select:focus { border-color: var(--accent); }
  .field-select { appearance: none; cursor: pointer; }
  option { background: var(--bg3); }
  .field-input-static {
    background: var(--bg4); border: 1px solid var(--border);
    color: var(--text3); font-family: 'Share Tech Mono', monospace; font-size: 0.9rem;
    padding: 0.6rem 0.9rem; border-radius: 2px; width: 100%;
    cursor: default; user-select: none; letter-spacing: 0.05em;
  }
  .wet-toggle {
    display: flex; align-items: center; gap: 8px; cursor: pointer;
    padding: 0.45rem 0.9rem;
    background: var(--bg3); border: 1px solid var(--border2); border-radius: 2px;
    font-family: 'Share Tech Mono', monospace; font-size: 0.72rem; letter-spacing: 0.08em;
    color: var(--text3); transition: all 0.2s; user-select: none;
  }
  .wet-toggle.active { border-color: #0067FF; color: #0067FF; background: rgba(0,103,255,0.08); }
  .wet-toggle-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--border2); transition: background 0.2s; }
  .wet-toggle.active .wet-toggle-dot { background: #0067FF; box-shadow: 0 0 6px #0067FF; }

  /* ── BUTTONS ── */
  .btn {
    font-family: 'Rajdhani', sans-serif; font-weight: 700; font-size: 0.9rem;
    letter-spacing: 0.1em; text-transform: uppercase;
    padding: 0.7rem 1.6rem; border: none; border-radius: 2px; cursor: pointer;
    transition: all 0.15s; position: relative; overflow: hidden;
  }
  .btn-primary {
    background: var(--red); color: #fff;
    clip-path: polygon(0 0, calc(100% - 12px) 0, 100% 100%, 12px 100%);
    padding: 0.7rem 2rem;
  }
  .btn-primary:hover { background: #ff1040; transform: translateY(-1px); }
  .btn-primary:active { transform: translateY(0); }
  .btn-primary:disabled { background: var(--bg4); color: var(--text3); cursor: not-allowed; transform: none; }
  .btn-secondary {
    background: var(--bg4); color: var(--text2); border: 1px solid var(--border2);
  }
  .btn-secondary:hover { border-color: var(--accent); color: var(--text); }
  .btn-ghost {
    background: transparent; color: var(--text3); border: 1px solid var(--border);
    font-size: 0.8rem; padding: 0.4rem 0.8rem;
  }
  .btn-ghost:hover { color: var(--text); border-color: var(--border2); }
  .btn-danger {
    background: transparent; color: var(--red); border: 1px solid var(--red-dim);
    font-size: 0.8rem; padding: 0.4rem 0.8rem;
  }

  /* ── COMPOUND BADGE ── */
  .compound-badge {
    display: inline-block; padding: 0.15rem 0.6rem;
    font-family: 'Share Tech Mono', monospace; font-size: 0.72rem; letter-spacing: 0.08em;
    border-radius: 2px; font-weight: 700;
  }

  /* ── PIT STOP ROW ── */
  .pit-row {
    display: grid; grid-template-columns: 80px 1fr 1fr auto; gap: 0.75rem; align-items: center;
    padding: 0.75rem 1rem; background: var(--bg3); border: 1px solid var(--border);
    border-radius: 2px; margin-bottom: 0.5rem;
  }
  .pit-number {
    font-family: 'Orbitron', monospace; font-size: 0.85rem; font-weight: 700;
    color: var(--text3); text-align: center;
  }
  .pit-number span { color: var(--red); font-size: 1.1rem; }

  /* ── STINT VISUALIZER ── */
  .stint-bar { display: flex; width: 100%; height: 32px; border-radius: 2px; overflow: hidden; gap: 1px; }
  .stint-segment {
    display: flex; align-items: center; justify-content: center;
    font-family: 'Share Tech Mono', monospace; font-size: 0.65rem; font-weight: 700;
    letter-spacing: 0.05em; color: rgba(0,0,0,0.8);
    transition: all 0.3s; min-width: 0;
    position: relative; overflow: hidden;
  }
  .stint-segment:hover { filter: brightness(1.15); }

  /* ── RESULT CARDS ── */
  .result-block {
    background: var(--bg3); border: 1px solid var(--border);
    padding: 1.2rem 1.5rem; border-radius: 2px;
    position: relative; overflow: hidden;
  }
  .result-block.optimal::before {
    content: ''; position: absolute; inset: 0;
    background: linear-gradient(135deg, rgba(0,200,81,0.05), transparent);
    pointer-events: none;
  }
  .result-label {
    font-family: 'Share Tech Mono', monospace; font-size: 0.7rem;
    letter-spacing: 0.12em; text-transform: uppercase; color: var(--text3);
    margin-bottom: 0.8rem; display: flex; align-items: center; gap: 6px;
  }
  .result-label .dot { width: 6px; height: 6px; border-radius: 50%; }
  .result-time {
    font-family: 'Orbitron', monospace; font-size: 2rem; font-weight: 700;
    color: var(--text); letter-spacing: 0.04em;
  }
  .result-time span { font-size: 1.1rem; color: var(--text3); margin-left: 4px; }
  .result-pos {
    font-family: 'Orbitron', monospace; font-size: 1rem; font-weight: 500; color: var(--text2);
    margin-top: 0.3rem;
  }
  .result-pos .pos-num { color: var(--gold); font-size: 1.4rem; font-weight: 700; }
  .delta-badge {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 0.2rem 0.6rem; border-radius: 2px;
    font-family: 'Share Tech Mono', monospace; font-size: 0.8rem; font-weight: 700;
    margin-top: 0.5rem;
  }
  .delta-badge.better { background: rgba(0,200,81,0.15); color: var(--green); }
  .delta-badge.worse  { background: rgba(232,0,45,0.15);  color: var(--red);   }
  .delta-badge.same   { background: rgba(255,215,0,0.1);  color: var(--gold);  }

  /* ── VERDICT BANNER ── */
  .verdict-banner {
    background: linear-gradient(90deg, var(--bg4), var(--bg3));
    border: 1px solid var(--border2); border-left: 3px solid var(--green);
    padding: 1rem 1.5rem; border-radius: 2px;
    font-family: 'Rajdhani', sans-serif; font-size: 1.05rem; font-weight: 600;
    color: var(--text); letter-spacing: 0.04em;
  }

  /* ── TELEMETRY LINE ── */
  .telem-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.45rem 0; border-bottom: 1px solid rgba(30,42,53,0.6);
  }
  .telem-row:last-child { border-bottom: none; }
  .telem-key { font-family: 'Share Tech Mono', monospace; font-size: 0.72rem; color: var(--text3); letter-spacing: 0.06em; }
  .telem-val { font-family: 'Rajdhani', monospace; font-size: 0.9rem; font-weight: 600; color: var(--text2); }
  .telem-val.highlight { color: var(--accent); }

  /* ── STANDINGS TABLE ── */
  .standings-table { width: 100%; border-collapse: collapse; }
  .standings-table th {
    font-family: 'Share Tech Mono', monospace; font-size: 0.68rem; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--text3);
    padding: 0.6rem 0.8rem; text-align: left;
    border-bottom: 1px solid var(--border2); background: var(--bg3);
  }
  .standings-table td {
    padding: 0.6rem 0.8rem; border-bottom: 1px solid var(--border);
    font-size: 0.92rem; font-weight: 500; color: var(--text2);
    vertical-align: middle;
  }
  .standings-table tr:hover td { background: rgba(30,42,53,0.4); }
  .standings-table tr:first-child td { color: var(--gold); }
  .standings-table tr:nth-child(2) td { color: #C0C0C0; }
  .standings-table tr:nth-child(3) td { color: #CD7F32; }
  .pos-col { font-family: 'Orbitron', monospace; font-weight: 700; font-size: 0.9rem; color: var(--text3) !important; width: 40px; }
  .pts-col { font-family: 'Orbitron', monospace; font-weight: 700; text-align: right !important; }
  .team-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
  .driver-code { font-family: 'Orbitron', monospace; font-weight: 700; font-size: 0.85rem; }

  /* ── DRIVER CARD ── */
  .driver-card {
    background: var(--bg2); border: 1px solid var(--border);
    border-radius: 2px; padding: 0.9rem 1rem; cursor: pointer;
    transition: all 0.15s; position: relative; overflow: hidden;
  }
  .driver-card:hover { border-color: var(--border2); transform: translateY(-1px); }
  .driver-card.selected { border-color: var(--red); }
  .driver-card::before {
    content: ''; position: absolute; top: 0; left: 0; bottom: 0; width: 3px;
  }
  .driver-card-num {
    font-family: 'Orbitron', monospace; font-size: 0.65rem; color: var(--text3); margin-bottom: 4px;
  }
  .driver-card-code {
    font-family: 'Orbitron', monospace; font-weight: 700; font-size: 1rem; margin-bottom: 2px;
  }
  .driver-card-name { font-size: 0.82rem; color: var(--text3); font-weight: 500; }
  .driver-card-team { font-size: 0.75rem; color: var(--text3); margin-top: 4px; font-weight: 400; }
  .driver-card-pace {
    font-family: 'Share Tech Mono', monospace; font-size: 0.7rem; color: var(--text3);
    margin-top: 6px;
  }

  /* ── CALENDAR ── */
  .calendar-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 0.6rem; }
  .race-card {
    background: var(--bg2); border: 1px solid var(--border);
    border-radius: 2px; padding: 0.75rem 1rem; position: relative; overflow: hidden;
  }
  .race-card.completed { opacity: 0.5; }
  .race-card.next { border-color: var(--red); }
  .race-card.next::before { content: ''; position: absolute; top:0;left:0;right:0; height:2px; background: var(--red); }
  .race-round { font-family: 'Share Tech Mono', monospace; font-size: 0.65rem; color: var(--text3); }
  .race-country { font-family: 'Orbitron', monospace; font-size: 0.8rem; font-weight: 600; color: var(--text); margin-top: 2px; }
  .race-circuit { font-size: 0.78rem; color: var(--text3); margin-top: 2px; }
  .race-date { font-family: 'Share Tech Mono', monospace; font-size: 0.65rem; color: var(--text3); margin-top: 6px; }
  .race-status-badge {
    display: inline-block; padding: 0.1rem 0.4rem; border-radius: 2px;
    font-family: 'Share Tech Mono', monospace; font-size: 0.62rem; margin-top: 4px;
  }
  .race-status-badge.next { background: rgba(232,0,45,0.15); color: var(--red); }
  .race-status-badge.upcoming { background: rgba(0,168,232,0.1); color: var(--accent); }
  .race-status-badge.completed { background: rgba(0,200,81,0.1); color: var(--green); }

  /* ── LOADING ── */
  .spinner {
    display: inline-block; width: 18px; height: 18px;
    border: 2px solid var(--border2); border-top-color: var(--red);
    border-radius: 50%; animation: spin 0.6s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ── ERROR / INFO ── */
  .alert {
    padding: 0.8rem 1.2rem; border-radius: 2px;
    font-family: 'Rajdhani', sans-serif; font-size: 0.9rem; font-weight: 500;
  }
  .alert-error { background: rgba(232,0,45,0.1); border: 1px solid rgba(232,0,45,0.3); color: #ff6080; }
  .alert-info  { background: rgba(0,168,232,0.08); border: 1px solid rgba(0,168,232,0.2); color: var(--accent); }
  .alert-success { background: rgba(0,200,81,0.08); border: 1px solid rgba(0,200,81,0.25); color: var(--green); }

  /* ── RESPONSIVE ── */
  @media (max-width: 768px) {
    .grid-2, .grid-3 { grid-template-columns: 1fr; }
    .header { padding: 0 1rem; }
    .main { padding: 1rem; }
    .pit-row { grid-template-columns: 60px 1fr 1fr auto; }
  }

  /* ── MISC ── */
  .mb-1 { margin-bottom: 0.5rem; }
  .mb-2 { margin-bottom: 1rem; }
  .mb-3 { margin-bottom: 1.5rem; }
  .mt-1 { margin-top: 0.5rem; }
  .mt-2 { margin-top: 1rem; }
  .flex { display: flex; }
  .gap-1 { gap: 0.5rem; }
  .gap-2 { gap: 1rem; }
  .items-center { align-items: center; }
  .justify-between { justify-content: space-between; }
  .flex-wrap { flex-wrap: wrap; }
  .text-center { text-align: center; }
  .w-full { width: 100%; }
  .separator { border: none; border-top: 1px solid var(--border); margin: 1.5rem 0; }
  .mono { font-family: 'Share Tech Mono', monospace; }
  .label-small { font-family: 'Share Tech Mono', monospace; font-size: 0.68rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--text3); }
  .testing-rank { font-family: 'Orbitron', monospace; font-size: 0.7rem; color: var(--text3); }
  .gap-badge {
    display: inline-block; padding: 0.1rem 0.5rem; border-radius: 2px;
    font-family: 'Share Tech Mono', monospace; font-size: 0.7rem;
    background: rgba(0,168,232,0.1); color: var(--accent);
  }
  .section-tabs { display: flex; gap: 1px; margin-bottom: 1.5rem; background: var(--bg3); padding: 3px; border-radius: 2px; width: fit-content; }
  .section-tab {
    padding: 0.45rem 1.1rem; border-radius: 1px; border: none;
    font-family: 'Rajdhani', sans-serif; font-weight: 600; font-size: 0.8rem;
    letter-spacing: 0.06em; text-transform: uppercase; cursor: pointer;
    color: var(--text3); background: transparent; transition: all 0.15s;
  }
  .section-tab.active { background: var(--bg2); color: var(--text); }
  .empty-state { text-align: center; padding: 3rem; color: var(--text3); font-family: 'Share Tech Mono', monospace; font-size: 0.8rem; letter-spacing: 0.08em; }
  .pace-bar-wrap { display: flex; align-items: center; gap: 8px; }
  .pace-bar { height: 4px; border-radius: 1px; background: var(--border2); flex: 1; overflow: hidden; }
  .pace-bar-fill { height: 100%; border-radius: 1px; transition: width 0.5s ease; }
  .tick-list { list-style: none; }
  .tick-list li { padding: 0.3rem 0; font-size: 0.88rem; color: var(--text2); display: flex; align-items: flex-start; gap: 8px; }
  .tick-list li::before { content: '▸'; color: var(--red); font-size: 0.7rem; margin-top: 2px; flex-shrink: 0; }
`;

// ─── COMPOUND BADGE ────────────────────────────────────────────────────────────
function CompoundBadge({ compound }) {
  const color = COMPOUND_COLORS[compound] || "#888";
  const bg    = compound === "HARD" ? "rgba(255,255,255,0.1)" :
                compound === "MEDIUM" ? "rgba(255,242,0,0.15)" :
                `${color}22`;
  return (
    <span className="compound-badge" style={{ background: bg, color, border: `1px solid ${color}44` }}>
      {compound}
    </span>
  );
}

// ─── STINT BAR ────────────────────────────────────────────────────────────────
function StintBar({ stints, totalLaps }) {
  if (!stints?.length) return null;
  return (
    <div className="stint-bar">
      {stints.map((s, i) => {
        const pct = ((s.laps / totalLaps) * 100).toFixed(1);
        const color = COMPOUND_COLORS[s.compound] || "#888";
        return (
          <div
            key={i}
            className="stint-segment"
            style={{ width: `${pct}%`, background: color, color: color === "#FFFFFF" ? "#111" : "#000" }}
            title={`${s.compound} · Laps ${s.lap_start}–${s.lap_end} (${s.laps} laps)`}
          >
            {s.laps > (totalLaps * 0.08) ? `${s.compound[0]} ${s.laps}L` : ""}
          </div>
        );
      })}
    </div>
  );
}

// ─── TELEM ROW ────────────────────────────────────────────────────────────────
function TelemRow({ label, value, highlight }) {
  return (
    <div className="telem-row">
      <span className="telem-key">{label}</span>
      <span className={`telem-val ${highlight ? "highlight" : ""}`}>{value}</span>
    </div>
  );
}

// ─── LOADING OVERLAY ──────────────────────────────────────────────────────────
function Spinner({ label }) {
  return (
    <div className="flex items-center gap-2" style={{ color: "var(--text3)", fontFamily: "'Share Tech Mono'" }}>
      <div className="spinner" />
      <span style={{ fontSize: "0.8rem", letterSpacing: "0.08em" }}>{label || "PROCESSING…"}</span>
    </div>
  );
}

// ─── API HOOK ─────────────────────────────────────────────────────────────────
function useApiStatus() {
  const [status, setStatus] = useState("checking");
  useEffect(() => {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 3000);
    fetch(`${API_BASE}/docs`, { signal: controller.signal, mode: "no-cors" })
      .then(() => { clearTimeout(timer); setStatus("online"); })
      .catch(e => { clearTimeout(timer); setStatus(e.name === "AbortError" ? "offline" : "online"); });
  }, []);
  return status;
}

// ═══════════════════════════════════════════════════════════════════════════════
// TAB: STRATEGY SIMULATOR (Phase 2 — /simulate)
// ═══════════════════════════════════════════════════════════════════════════════
function StrategySimulator() {
  const YEARS = [2021, 2022, 2023, 2024, 2025];

  const [form, setForm] = useState({
    year: 2024,
    gp_name: "Spanish Grand Prix",
    driver_code: "VER",
    starting_compound: "SOFT",
  });
  const [pitStops, setPitStops] = useState([{ lap: "20", compound: "HARD" }]);
  const [wetRace, setWetRace]   = useState(false);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState(null);
  const [result, setResult]     = useState(null);

  const totalLaps = CIRCUIT_LAPS[form.gp_name] || 57;
  // All 5 compounds always available — wet races commonly transition to slicks
  // as the track dries. The only constraint: a wet race must include at least
  // one INTERMEDIATE or WET compound somewhere in the strategy.
  const availableCompounds = COMPOUNDS;

  const addPit = () => {
    if (pitStops.length >= 3) return;
    const defaultLap = Math.round(totalLaps * (pitStops.length === 0 ? 0.35 : pitStops.length === 1 ? 0.65 : 0.80));
    setPitStops([...pitStops, { lap: String(defaultLap), compound: "MEDIUM" }]);
  };
  const removePit = (i) => setPitStops(pitStops.filter((_, idx) => idx !== i));
  // Store lap as raw string — user can clear and retype freely; clamping on blur + submit
  const updatePit = (i, field, val) => {
    const next = [...pitStops];
    next[i] = { ...next[i], [field]: val };
    setPitStops(next);
  };
  const blurPit = (i) => {
    const next = [...pitStops];
    const clamped = Math.min(Math.max(1, parseInt(next[i].lap) || 1), totalLaps - 1);
    next[i] = { ...next[i], lap: String(clamped) };
    setPitStops(next);
  };

  // When wet toggle changes, only adjust the starting compound as a helpful
  // default. Existing pit stop compounds are left as-is so the user can model
  // a drying-track strategy (e.g. INTER start → SOFT finish).
  const handleWetToggle = () => {
    const nowWet = !wetRace;
    setWetRace(nowWet);
    if (nowWet) {
      // Default start to INTERMEDIATE if currently on a dry compound
      setForm(f => ({
        ...f,
        starting_compound: !["INTERMEDIATE","WET"].includes(f.starting_compound) ? "INTERMEDIATE" : f.starting_compound,
      }));
    } else {
      // Back to dry — if starting on a wet compound, swap it to SOFT
      setForm(f => ({
        ...f,
        starting_compound: ["INTERMEDIATE","WET"].includes(f.starting_compound) ? "SOFT" : f.starting_compound,
      }));
    }
  };

  const historicalDrivers = [
    "VER","HAM","LEC","NOR","SAI","RUS","PER","ALO","STR","GAS","ALB",
    "OCO","BEA","HUL","LAW","RIC","ZHO","BOT","MAG","TSU","COL","PIA",
  ];

  const simulate = async () => {
    // Client-side compound guard
    const WET_C = ["INTERMEDIATE","WET"];
    const DRY_C = ["SOFT","MEDIUM","HARD"];
    const allC  = [form.starting_compound, ...pitStops.map(p => p.compound)];
    if (wetRace) {
      // Must have at least one wet compound — but dry slicks are fine too (drying track)
      if (!allC.some(c => WET_C.includes(c))) { setError("Wet race: strategy must include at least one INTERMEDIATE or WET compound."); return; }
    } else {
      if (allC.some(c => WET_C.includes(c))) { setError("Dry race: cannot use INTERMEDIATE or WET compounds."); return; }
    }

    setLoading(true); setError(null); setResult(null);
    try {
      const sanitisedPits = pitStops.map(ps => ({
        ...ps,
        lap: Math.min(Math.max(1, parseInt(ps.lap) || 1), totalLaps - 1),
      }));
      const res = await fetch(`${API_BASE}/simulate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          year: parseInt(form.year),
          gp_name: form.gp_name,
          driver_code: form.driver_code.toUpperCase(),
          starting_compound: form.starting_compound,
          weather_code: wetRace ? 3.0 : 1.0,
          pit_stops: sanitisedPits,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        if (Array.isArray(data.detail)) {
          throw new Error(data.detail.map(e => e.msg || JSON.stringify(e)).join(" · "));
        }
        throw new Error(data.detail || "API error");
      }
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const formatTime = (t) => {
    if (!t) return "—";
    const m = Math.floor(t / 60);
    const s = (t % 60).toFixed(3);
    return `${m}:${s.padStart(6, "0")}`;
  };

  const posOrdinal = (n) => {
    if (!n) return "—";
    return `P${n}`;
  };

  return (
    <div>
      <div className="section-head">
        <div className="section-title">Strategy Simulator</div>
        <div className="section-sub">Simulate historical race strategies and find the optimal pit window · 2021–2025</div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "minmax(0,2fr) minmax(0,3fr)", gap: "1.5rem", alignItems: "start" }}>
        {/* FORM */}
        <div className="panel">
          <div className="panel-header">⬡ Strategy Input</div>
          <div className="panel-body">
            <div className="grid-2 mb-2">
              <div className="field">
                <label className="field-label">Season</label>
                <select className="field-select" value={form.year}
                  onChange={e => setForm({ ...form, year: e.target.value })}>
                  {YEARS.map(y => <option key={y} value={y}>{y}</option>)}
                </select>
              </div>
              <div className="field">
                <label className="field-label">Driver Code</label>
                <select className="field-select" value={form.driver_code}
                  onChange={e => setForm({ ...form, driver_code: e.target.value })}>
                  {historicalDrivers.map(d => <option key={d} value={d}>{d}</option>)}
                </select>
              </div>
            </div>

            <div className="grid-2 mb-2">
              <div className="field">
                <label className="field-label">Grand Prix</label>
                <select className="field-select" value={form.gp_name}
                  onChange={e => setForm({ ...form, gp_name: e.target.value })}>
                  {HISTORICAL_CIRCUITS.sort().map(name => (
                    <option key={name} value={name}>{name}</option>
                  ))}
                </select>
              </div>
              <div className="field">
                <label className="field-label">Race Laps</label>
                <div className="field-input-static">{totalLaps} laps</div>
              </div>
            </div>

            <div className="field mb-3">
              <div className="flex items-center justify-between mb-1">
                <label className="field-label">Starting Compound</label>
                <div className={`wet-toggle ${wetRace ? "active" : ""}`} onClick={handleWetToggle}>
                  <div className="wet-toggle-dot" />
                  WET RACE
                </div>
              </div>
              <div className="flex gap-1 flex-wrap">
                {availableCompounds.map(c => (
                  <button key={c} className={`btn ${form.starting_compound === c ? "btn-primary" : "btn-secondary"}`}
                    style={{
                      padding: "0.45rem 1rem", fontSize: "0.8rem",
                      ...(form.starting_compound === c ? { background: COMPOUND_COLORS[c], color: c === "MEDIUM" ? "#000" : c === "HARD" ? "#000" : "#fff" } : {}),
                    }}
                    onClick={() => setForm({ ...form, starting_compound: c })}>
                    {c}
                  </button>
                ))}
              </div>
            </div>

            <div className="mb-2">
              <div className="flex items-center justify-between mb-1">
                <span className="label-small">Pit Stops ({pitStops.length}/3)</span>
                {pitStops.length < 3 && (
                  <button className="btn btn-ghost" onClick={addPit}>+ Add Stop</button>
                )}
              </div>
              {pitStops.length === 0 && (
                <div className="alert alert-info" style={{ fontSize: "0.82rem" }}>No pit stops — one compound strategy.</div>
              )}
              {pitStops.map((ps, i) => (
                <div className="pit-row" key={i}>
                  <div className="pit-number">PIT <span>{i + 1}</span></div>
                  <div className="field">
                    <label className="field-label">Lap</label>
                    <input className="field-input" type="number" value={ps.lap}
                      onChange={e => updatePit(i, "lap", e.target.value)}
                      onBlur={() => blurPit(i)} />
                  </div>
                  <div className="field">
                    <label className="field-label">Compound</label>
                    <select className="field-select" value={ps.compound}
                      onChange={e => updatePit(i, "compound", e.target.value)}>
                      {availableCompounds.map(c => <option key={c} value={c}>{c}</option>)}
                    </select>
                  </div>
                  <button className="btn btn-danger" onClick={() => removePit(i)}>✕</button>
                </div>
              ))}
            </div>

            <button className="btn btn-primary w-full" onClick={simulate} disabled={loading}
              style={{ justifyContent: "center", display: "flex", alignItems: "center", gap: 10 }}>
              {loading ? <><Spinner /> Simulating…</> : "⚡ Run Simulation"}
            </button>

            {error && <div className="alert alert-error mt-1">{error}</div>}
          </div>
        </div>

        {/* RESULTS */}
        <div>
          {!result && !loading && (
            <div className="panel" style={{ height: "100%" }}>
              <div className="empty-state">
                <div style={{ fontSize: "2rem", marginBottom: "0.5rem" }}>◈</div>
                <div>AWAITING SIMULATION INPUT</div>
                <div style={{ marginTop: "0.5rem", color: "var(--text3)", fontSize: "0.72rem" }}>
                  Configure strategy and run simulation
                </div>
              </div>
            </div>
          )}
          {loading && (
            <div className="panel" style={{ height: "100%" }}>
              <div className="empty-state"><Spinner label="RUNNING NEURAL INFERENCE…" /></div>
            </div>
          )}
          {result && (
            <div>
              {/* Race Header */}
              <div className="panel mb-2">
                <div className="panel-body" style={{ paddingBottom: "1rem" }}>
                  <div className="flex items-center justify-between flex-wrap gap-2">
                    <div>
                      <div style={{ fontFamily: "'Orbitron'", fontSize: "0.7rem", color: "var(--text3)", letterSpacing: "0.12em" }}>
                        {result.year} · ROUND {result.gp_name}
                      </div>
                      <div style={{ fontFamily: "'Orbitron'", fontSize: "1.1rem", fontWeight: 700, marginTop: 4 }}>
                        {result.driver_code} — {result.team_name}
                      </div>
                    </div>
                    <div style={{ textAlign: "right" }}>
                      <div className="label-small">Grid Position</div>
                      <div style={{ fontFamily: "'Orbitron'", fontSize: "1.4rem", fontWeight: 700, color: "var(--gold)" }}>
                        P{result.starting_grid_position}
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Comparison */}
              <div className="grid-2 mb-2">
                <div className="result-block">
                  <div className="result-label"><span className="dot" style={{ background: "var(--accent)" }} />Your Strategy</div>
                  <div className="result-time">
                    {formatTime(result.user_strategy.predicted_avg_lap_time)}<span>avg</span>
                  </div>
                  <div className="result-pos">
                    Finish: <span className="pos-num">{posOrdinal(result.user_strategy.estimated_finishing_position)}</span>
                  </div>
                  <div style={{ marginTop: 8 }}>
                    <StintBar stints={result.user_strategy.stints} totalLaps={result.race_laps} />
                  </div>
                  <div className="mt-1">
                    <TelemRow label="Pit Stops" value={result.user_strategy.num_pit_stops} />
                    <TelemRow label="Actual Avg" value={formatTime(result.user_strategy.actual_avg_lap_time)} />
                  </div>
                </div>

                <div className="result-block optimal">
                  <div className="result-label"><span className="dot" style={{ background: "var(--green)" }} />Optimal Strategy</div>
                  <div className="result-time">
                    {formatTime(result.optimal_strategy.predicted_avg_lap_time)}<span>avg</span>
                  </div>
                  <div className="result-pos">
                    Finish: <span className="pos-num">{posOrdinal(result.optimal_strategy.estimated_finishing_position)}</span>
                  </div>
                  <div style={{ marginTop: 8 }}>
                    <StintBar stints={result.optimal_strategy.stints} totalLaps={result.race_laps} />
                  </div>
                  <div className="mt-1">
                    <TelemRow label="Pit Stops" value={result.optimal_strategy.num_pit_stops} />
                    <TelemRow label="Strategies Tested" value={result.optimal_strategy.strategies_evaluated || "—"} />
                  </div>
                </div>
              </div>

              {/* Delta */}
              <div style={{ marginBottom: "0.75rem" }}>
                {(() => {
                  const d = result.time_delta_per_lap;
                  return (
                    <span className={`delta-badge ${d > 0 ? "worse" : d < 0 ? "better" : "same"}`}>
                      {d > 0 ? "▲" : d < 0 ? "▼" : "●"} {Math.abs(d).toFixed(3)}s/lap vs optimal
                    </span>
                  );
                })()}
              </div>

              {/* Verdict */}
              <div className="verdict-banner">{result.verdict}</div>

              {/* Optimal Strategy Details */}
              {result.optimal_strategy.pit_stops?.length > 0 && (
                <div className="panel mt-2">
                  <div className="panel-header">◈ Optimal Pit Strategy</div>
                  <div className="panel-body">
                    <div className="flex items-center gap-2 mb-2">
                      <span className="label-small">Start:</span>
                      <CompoundBadge compound={result.optimal_strategy.starting_compound} />
                    </div>
                    {result.optimal_strategy.pit_stops.map((ps, i) => (
                      <div key={i} className="telem-row">
                        <span className="telem-key">Pit {i + 1} — Lap {ps.lap}</span>
                        <CompoundBadge compound={ps.compound} />
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// TAB: 2026 PREDICTOR
// ═══════════════════════════════════════════════════════════════════════════════
function Predictor2026() {
  const [selectedDriver, setSelectedDriver] = useState(null);
  const [form, setForm] = useState({
    gp_name: "Australian Grand Prix",
    starting_compound: "SOFT",
    grid_position: 10,
    weather_code: 1.0,
    race_number: 1,
  });
  const [pitStops, setPitStops] = useState([{ lap: "20", compound: "HARD" }]);
  const [wetRace, setWetRace]   = useState(false);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState(null);
  const [result, setResult]     = useState(null);

  const totalLaps = CIRCUIT_LAPS[form.gp_name] || 57;
  // All 5 compounds always available — wet races commonly transition to slicks
  // as the track dries. Only constraint: wet race must include at least one
  // INTERMEDIATE or WET compound somewhere in the strategy.
  const availableCompounds = COMPOUNDS;

  const handleGpChange = (gpName) => {
    const circuit = CIRCUITS_2026.find(c => c.name === gpName);
    setForm(f => ({
      ...f,
      gp_name: gpName,
      race_number: circuit ? circuit.round : f.race_number,
    }));
  };

  const handleWetToggle = () => {
    const nowWet = !wetRace;
    setWetRace(nowWet);
    if (nowWet) {
      // Default start to INTERMEDIATE if on a dry compound; leave pit stop
      // compounds untouched so users can model drying-track transitions freely.
      setForm(f => ({
        ...f,
        weather_code: 3.0,
        starting_compound: !["INTERMEDIATE","WET"].includes(f.starting_compound) ? "INTERMEDIATE" : f.starting_compound,
      }));
    } else {
      // Back to dry — swap starting compound if it's a wet tyre
      setForm(f => ({
        ...f,
        weather_code: 1.0,
        starting_compound: ["INTERMEDIATE","WET"].includes(f.starting_compound) ? "SOFT" : f.starting_compound,
      }));
    }
  };

  const addPit = () => {
    if (pitStops.length < 3) {
      const defaultLap = Math.round(totalLaps * (pitStops.length === 0 ? 0.35 : pitStops.length === 1 ? 0.65 : 0.80));
      setPitStops([...pitStops, { lap: String(defaultLap), compound: "MEDIUM" }]);
    }
  };
  const removePit = (i) => setPitStops(pitStops.filter((_, idx) => idx !== i));
  const updatePit = (i, f, v) => {
    const n = [...pitStops];
    n[i] = { ...n[i], [f]: v };
    setPitStops(n);
  };
  const blurPit = (i) => {
    const n = [...pitStops];
    const clamped = Math.min(Math.max(1, parseInt(n[i].lap) || 1), totalLaps - 1);
    n[i] = { ...n[i], lap: String(clamped) };
    setPitStops(n);
  };

  const selectedDriverInfo = DRIVERS_2026.find(d => d.code === selectedDriver);

  const predict = async () => {
    if (!selectedDriver) { setError("Select a driver first."); return; }

    // Client-side compound guard (backend also validates)
    const WET_COMPOUNDS = ["INTERMEDIATE", "WET"];
    const allCompounds  = [form.starting_compound, ...pitStops.map(p => p.compound)];
    if (wetRace) {
      // Must include at least one wet compound — dry slicks are fine too (drying track)
      if (!allCompounds.some(c => WET_COMPOUNDS.includes(c))) {
        setError("Wet race: strategy must include at least one INTERMEDIATE or WET compound."); return;
      }
    } else {
      const hasWet = allCompounds.some(c => WET_COMPOUNDS.includes(c));
      if (hasWet) { setError("Dry race: cannot use INTERMEDIATE or WET compounds."); return; }
    }

    setLoading(true); setError(null); setResult(null);
    try {
      const sanitisedPits = pitStops.map(ps => ({
        ...ps,
        lap: Math.min(Math.max(1, parseInt(ps.lap) || 1), totalLaps - 1),
      }));
      const res = await fetch(`${API_BASE}/2026/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          gp_name: form.gp_name,
          driver_code: selectedDriver,
          team_name: selectedDriverInfo?.team || "",
          starting_compound: form.starting_compound,
          pit_stops: sanitisedPits,
          grid_position: parseInt(form.grid_position),
          total_laps: totalLaps,
          weather_code: parseFloat(form.weather_code),
          race_number: parseInt(form.race_number),
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        if (Array.isArray(data.detail)) {
          throw new Error(data.detail.map(e => e.msg || JSON.stringify(e)).join(" · "));
        }
        throw new Error(data.detail || "API error");
      }
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const formatTime = (t) => {
    if (!t) return "—";
    const m = Math.floor(t / 60);
    const s = (t % 60).toFixed(3);
    return `${m}:${s.padStart(6, "0")}`;
  };

  const teamColor = selectedDriverInfo ? (TEAM_COLORS[selectedDriverInfo.team] || "#888") : "var(--red)";

  return (
    <div>
      <div className="section-head">
        <div className="section-title">2026 Season Predictor</div>
        <div className="section-sub">Neural model with 2026 regulation scaling · Adapted from 2021–2025 training data</div>
      </div>

      {/* Driver Grid */}
      <div className="panel mb-3">
        <div className="panel-header">⬡ Select Driver — 2026 Grid</div>
        <div className="panel-body">
          <div className="grid-auto">
            {DRIVERS_2026.map(d => {
              const tc = TEAM_COLORS[d.team] || "#888";
              return (
                <div
                  key={d.code}
                  className={`driver-card ${selectedDriver === d.code ? "selected" : ""}`}
                  onClick={() => { setSelectedDriver(d.code); setForm(f => ({ ...f, team_name: d.team })); }}
                  style={{ borderLeftColor: tc }}
                >
                  <div style={{ position: "absolute", top: 0, left: 0, bottom: 0, width: 3, background: tc }} />
                  <div className="driver-card-num">#{d.number}</div>
                  <div className="driver-card-code" style={{ color: selectedDriver === d.code ? tc : "var(--text)" }}>{d.code}</div>
                  <div className="driver-card-name">{d.name}</div>
                  <div className="driver-card-team">{d.team}</div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "minmax(0,2fr) minmax(0,3fr)", gap: "1.5rem", alignItems: "start" }}>
        {/* FORM */}
        <div className="panel">
          <div className="panel-header">
            ◈ Race Configuration
            {selectedDriverInfo && (
              <span style={{ marginLeft: "auto", color: teamColor, fontFamily: "'Rajdhani'", fontWeight: 600, fontSize: "0.85rem" }}>
                {selectedDriverInfo.code} · {selectedDriverInfo.team}
              </span>
            )}
          </div>
          <div className="panel-body">
            <div className="grid-2 mb-2">
              <div className="field">
                <label className="field-label">Grand Prix</label>
                <select className="field-select" value={form.gp_name}
                  onChange={e => handleGpChange(e.target.value)}>
                  {CIRCUITS_2026.map(c => <option key={c.name} value={c.name}>{c.name}</option>)}
                </select>
              </div>
              <div className="field">
                <label className="field-label">Race Round</label>
                <div className="field-input-static">R{form.race_number}</div>
              </div>
            </div>
            <div className="grid-2 mb-2">
              <div className="field">
                <label className="field-label">Grid Position</label>
                <input className="field-input" type="number" min="1" max="22" value={form.grid_position}
                  onChange={e => setForm({ ...form, grid_position: e.target.value })} />
              </div>
              <div className="field">
                <label className="field-label">Race Laps</label>
                <div className="field-input-static">{totalLaps} laps</div>
              </div>
            </div>

            <div className="field mb-3">
              <div className="flex items-center justify-between mb-1">
                <label className="field-label">Starting Compound</label>
                <div className={`wet-toggle ${wetRace ? "active" : ""}`} onClick={handleWetToggle}>
                  <div className="wet-toggle-dot" />
                  WET RACE
                </div>
              </div>
              <div className="flex gap-1 flex-wrap">
                {availableCompounds.map(c => (
                  <button key={c} className={`btn ${form.starting_compound === c ? "btn-primary" : "btn-secondary"}`}
                    style={{
                      padding: "0.45rem 1rem", fontSize: "0.8rem",
                      ...(form.starting_compound === c ? { background: COMPOUND_COLORS[c], color: c === "MEDIUM" ? "#000" : c === "HARD" ? "#000" : "#fff" } : {}),
                    }}
                    onClick={() => setForm({ ...form, starting_compound: c })}>
                    {c}
                  </button>
                ))}
              </div>
            </div>

            <div className="mb-3">
              <div className="flex items-center justify-between mb-1">
                <span className="label-small">Pit Stops ({pitStops.length}/3)</span>
                {pitStops.length < 3 && (
                  <button className="btn btn-ghost" onClick={addPit}>+ Add Stop</button>
                )}
              </div>
              {pitStops.map((ps, i) => (
                <div className="pit-row" key={i}>
                  <div className="pit-number">PIT <span>{i + 1}</span></div>
                  <div className="field">
                    <label className="field-label">Lap</label>
                    <input className="field-input" type="number" value={ps.lap}
                      onChange={e => updatePit(i, "lap", e.target.value)}
                      onBlur={() => blurPit(i)} />
                  </div>
                  <div className="field">
                    <label className="field-label">Compound</label>
                    <select className="field-select" value={ps.compound}
                      onChange={e => updatePit(i, "compound", e.target.value)}>
                      {availableCompounds.map(c => <option key={c} value={c}>{c}</option>)}
                    </select>
                  </div>
                  <button className="btn btn-danger" onClick={() => removePit(i)}>✕</button>
                </div>
              ))}
            </div>

            <button className="btn btn-primary w-full" onClick={predict} disabled={loading || !selectedDriver}
              style={{ justifyContent: "center", display: "flex", alignItems: "center", gap: 10 }}>
              {loading ? <><Spinner />Predicting…</> : "◈ Predict 2026"}
            </button>
            {error && <div className="alert alert-error mt-1">{error}</div>}
          </div>
        </div>

        {/* RESULT */}
        <div>
          {!result && !loading && (
            <div className="panel" style={{ minHeight: 300 }}>
              <div className="empty-state">
                <div style={{ fontSize: "2rem", marginBottom: "0.5rem" }}>◈</div>
                <div>{selectedDriver ? `${selectedDriver} SELECTED — CONFIGURE RACE` : "SELECT DRIVER TO BEGIN"}</div>
              </div>
            </div>
          )}
          {loading && (
            <div className="panel" style={{ minHeight: 300 }}>
              <div className="empty-state"><Spinner label="RUNNING 2026 PREDICTION…" /></div>
            </div>
          )}
          {result && (
            <div>
              {/* Header */}
              <div className="panel mb-2">
                <div className="panel-body">
                  <div className="flex items-center justify-between flex-wrap gap-2">
                    <div>
                      <div style={{ fontFamily: "'Orbitron'", fontSize: "0.7rem", color: "var(--text3)", letterSpacing: "0.12em" }}>
                        2026 · R{result.race_number} · {result.gp_name}
                      </div>
                      <div style={{ fontFamily: "'Orbitron'", fontSize: "1.1rem", fontWeight: 700, marginTop: 4, color: teamColor }}>
                        {result.driver_code} — {result.team_name}
                      </div>
                    </div>
                    <div style={{ textAlign: "right" }}>
                      <div className="label-small">Predicted Finish</div>
                      <div style={{ fontFamily: "'Orbitron'", fontSize: "1.8rem", fontWeight: 900, color: "var(--gold)" }}>
                        P{result.prediction.predicted_position}
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Lap Times */}
              <div className="grid-2 mb-2">
                <div className="result-block">
                  <div className="result-label">Raw Model Output</div>
                  <div className="result-time" style={{ fontSize: "1.5rem" }}>
                    {formatTime(result.prediction.avg_lap_time_raw)}
                  </div>
                  <div style={{ fontSize: "0.78rem", color: "var(--text3)", marginTop: 4 }}>Before 2026 regulation scaling</div>
                </div>
                <div className="result-block optimal">
                  <div className="result-label"><span className="dot" style={{ background: "var(--green)" }} />2026 Adjusted</div>
                  <div className="result-time" style={{ fontSize: "1.5rem" }}>
                    {formatTime(result.prediction.avg_lap_time_2026)}
                  </div>
                  <div style={{ fontSize: "0.78rem", color: "var(--text3)", marginTop: 4 }}>
                    Scaling: ×{result.prediction.pace_scaling?.toFixed(4)}
                  </div>
                </div>
              </div>

              {/* Strategy */}
              <div className="panel mb-2">
                <div className="panel-header">◈ Race Strategy</div>
                <div className="panel-body">
                  <StintBar stints={result.strategy.stints} totalLaps={totalLaps} />
                  <div className="mt-2">
                    <TelemRow label="Starting Compound" value={<CompoundBadge compound={result.strategy.starting_compound} />} />
                    <TelemRow label="Pit Stops" value={result.strategy.num_pit_stops} />
                    {result.strategy.pit_stops?.map((ps, i) => (
                      <TelemRow key={i} label={`Pit ${i + 1} — Lap ${ps.lap}`} value={<CompoundBadge compound={ps.compound} />} />
                    ))}
                    <TelemRow label="Year Proxy" value={result.prediction.year_proxy} />
                  </div>
                </div>
              </div>

              {/* Testing Context */}
              {result.testing_context && (
                <div className="panel mb-2">
                  <div className="panel-header">⚡ Bahrain Testing 2026</div>
                  <div className="panel-body">
                    <TelemRow label="Best Testing Lap" value={`${result.testing_context.testing_best}s`} highlight />
                    <TelemRow label="Testing Rank" value={`P${result.testing_context.testing_rank}`} />
                  </div>
                </div>
              )}

              {/* Adaptations */}
              {result.model_adaptations?.length > 0 && (
                <div className="panel">
                  <div className="panel-header">⬡ Model Adaptations</div>
                  <div className="panel-body">
                    <ul className="tick-list">
                      {result.model_adaptations.map((a, i) => (
                        <li key={i}>{a}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// TAB: 2026 SEASON HUB
// ═══════════════════════════════════════════════════════════════════════════════
function SeasonHub() {
  const nextRace = CIRCUITS_2026[0]; // Australian GP — Round 1

  return (
    <div>
      <div className="section-head">
        <div className="section-title">2026 Season Hub</div>
        <div className="section-sub">Live standings, calendar, and pace data — available once the season begins</div>
      </div>

      {/* Coming Soon Hero */}
      <div className="panel" style={{ marginBottom: "1.5rem", overflow: "hidden", position: "relative" }}>
        <div style={{
          position: "absolute", inset: 0, zIndex: 0,
          background: "radial-gradient(ellipse at 30% 50%, rgba(232,0,45,0.06) 0%, transparent 60%), radial-gradient(ellipse at 70% 50%, rgba(0,168,232,0.04) 0%, transparent 60%)",
          pointerEvents: "none",
        }} />
        <div className="panel-body" style={{ padding: "4rem 2rem", textAlign: "center", position: "relative", zIndex: 1 }}>
          <div style={{
            fontFamily: "'Orbitron', monospace", fontSize: "0.72rem", letterSpacing: "0.25em",
            color: "var(--red)", textTransform: "uppercase", marginBottom: "1.2rem",
          }}>
            ◈ &nbsp; Season begins · March 8, 2026 &nbsp; ◈
          </div>
          <div style={{
            fontFamily: "'Orbitron', monospace", fontSize: "clamp(1.8rem, 4vw, 3rem)",
            fontWeight: 900, letterSpacing: "0.08em", color: "var(--text)",
            lineHeight: 1.1, marginBottom: "0.6rem",
          }}>
            COMING SOON
          </div>
          <div style={{
            fontFamily: "'Rajdhani', sans-serif", fontSize: "1.05rem", fontWeight: 500,
            color: "var(--text3)", letterSpacing: "0.06em", marginBottom: "2.5rem",
          }}>
            Live standings, championship tracker, and Bayesian pace updates will unlock<br />
            after the first race result is entered
          </div>

          {/* Countdown pill */}
          <div style={{
            display: "inline-flex", alignItems: "center", gap: "1.5rem",
            background: "var(--bg3)", border: "1px solid var(--border2)",
            borderRadius: 2, padding: "0.9rem 2rem",
          }}>
            <div style={{ textAlign: "center" }}>
              <div style={{ fontFamily: "'Orbitron', monospace", fontSize: "1.6rem", fontWeight: 700, color: "var(--text)", lineHeight: 1 }}>R1</div>
              <div style={{ fontFamily: "'Share Tech Mono'", fontSize: "0.65rem", color: "var(--text3)", marginTop: 4, letterSpacing: "0.08em" }}>ROUND</div>
            </div>
            <div style={{ width: 1, height: 40, background: "var(--border2)" }} />
            <div style={{ textAlign: "center" }}>
              <div style={{ fontFamily: "'Orbitron', monospace", fontSize: "1.1rem", fontWeight: 700, color: "var(--gold)", lineHeight: 1 }}>AUS</div>
              <div style={{ fontFamily: "'Share Tech Mono'", fontSize: "0.65rem", color: "var(--text3)", marginTop: 4, letterSpacing: "0.08em" }}>ALBERT PARK</div>
            </div>
            <div style={{ width: 1, height: 40, background: "var(--border2)" }} />
            <div style={{ textAlign: "center" }}>
              <div style={{ fontFamily: "'Orbitron', monospace", fontSize: "1.1rem", fontWeight: 700, color: "var(--text)", lineHeight: 1 }}>08 MAR</div>
              <div style={{ fontFamily: "'Share Tech Mono'", fontSize: "0.65rem", color: "var(--text3)", marginTop: 4, letterSpacing: "0.08em" }}>2026</div>
            </div>
          </div>
        </div>
      </div>

      {/* What to expect */}
      <div className="grid-3">
        {[
          {
            icon: "🏆",
            title: "Live Standings",
            desc: "WDC and WCC tables update automatically after each race result is entered. Full points history, gap to leader, and mathematical elimination tracking.",
          },
          {
            icon: "⚡",
            title: "Bayesian Pace Updates",
            desc: "After each race, the model recalibrates team scaling factors based on actual vs predicted lap times — predictions improve as the season progresses.",
          },
          {
            icon: "📅",
            title: "Season Calendar",
            desc: "24-race calendar with circuit types, sprint weekends flagged, and race status tracking. Completed races link to their post-race insight reports.",
          },
        ].map((card, i) => (
          <div key={i} className="panel">
            <div className="panel-body">
              <div style={{ fontSize: "1.6rem", marginBottom: "0.75rem" }}>{card.icon}</div>
              <div style={{ fontFamily: "'Orbitron', monospace", fontSize: "0.8rem", fontWeight: 600, color: "var(--text)", marginBottom: "0.6rem", letterSpacing: "0.06em" }}>
                {card.title}
              </div>
              <div style={{ fontSize: "0.88rem", color: "var(--text3)", lineHeight: 1.6, fontWeight: 400 }}>
                {card.desc}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* 2026 Calendar Preview */}
      <div className="panel" style={{ marginTop: "1.5rem" }}>
        <div className="panel-header">◈ 2026 Race Calendar Preview</div>
        <div className="panel-body">
          <div className="calendar-grid">
            {CIRCUITS_2026.map((r, i) => (
              <div key={r.round} className={`race-card ${i === 0 ? "next" : "upcoming"}`}>
                <div className="race-round">ROUND {r.round}</div>
                <div className="race-country">{r.name.replace(" Grand Prix", "").replace(" City", "")}</div>
                <div className="race-circuit">{r.circuit}</div>
                <div>
                  <span className={`race-status-badge ${i === 0 ? "next" : "upcoming"}`}>
                    {i === 0 ? "▶ OPENS SEASON" : "○ UPCOMING"}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}


// ═══════════════════════════════════════════════════════════════════════════════
// ROOT APP
// ═══════════════════════════════════════════════════════════════════════════════
export default function App() {
  const [tab, setTab]   = useState("simulator");
  const apiStatus       = useApiStatus();

  const tabs = [
    { id: "simulator", label: "Strategy Simulator" },
    { id: "predict26", label: "2026 Predictor" },
    { id: "season26",  label: "Season Hub" },
  ];

  return (
    <>
      <style>{css}</style>
      <div className="header">
        <div className="header-logo">
          <div className="header-logo-mark" />
          F1 <span>STRATEGY</span> LAB
        </div>

        <nav className="nav-tabs">
          {tabs.map(t => (
            <button key={t.id} className={`nav-tab ${tab === t.id ? "active" : ""}`}
              onClick={() => setTab(t.id)}>{t.label}</button>
          ))}
        </nav>

        <div className="api-indicator">
          <div className={`api-dot ${apiStatus === "online" ? "online" : apiStatus === "offline" ? "offline" : ""}`} />
          API {apiStatus === "online" ? "ONLINE" : apiStatus === "offline" ? "OFFLINE" : "CHECKING"}
        </div>
      </div>

      <main className="main">
        {tab === "simulator" && <StrategySimulator />}
        {tab === "predict26" && <Predictor2026 />}
        {tab === "season26"  && <SeasonHub />}
      </main>
    </>
  );
}