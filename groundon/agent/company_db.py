"""Seed data for the fictional company "Pallet Dynamics" (SQLite)."""

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE employees (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    email       TEXT NOT NULL,
    department  TEXT NOT NULL,
    role        TEXT NOT NULL,
    location    TEXT NOT NULL,
    manager_id  INTEGER REFERENCES employees(id),
    hire_date   TEXT NOT NULL          -- ISO date
);
CREATE TABLE leave_balances (
    employee_id   INTEGER NOT NULL REFERENCES employees(id),
    year          INTEGER NOT NULL,
    annual_total  INTEGER NOT NULL,    -- includes carried-over days
    annual_used   INTEGER NOT NULL,
    sick_total    INTEGER NOT NULL,
    sick_used     INTEGER NOT NULL,
    PRIMARY KEY (employee_id, year)
);
CREATE TABLE it_tickets (
    id           INTEGER PRIMARY KEY,
    employee_id  INTEGER NOT NULL REFERENCES employees(id),
    title        TEXT NOT NULL,
    category     TEXT NOT NULL,        -- hardware | software | access | security
    priority     TEXT NOT NULL,        -- P1 | P2 | P3
    status       TEXT NOT NULL,        -- open | in_progress | resolved
    opened_at    TEXT NOT NULL,
    resolved_at  TEXT
);
"""

EMPLOYEES = [
    (1, "Ananya Rao", "ananya.rao@palletdynamics.example", "Engineering", "VP Engineering", "Bengaluru", None, "2018-04-02"),
    (2, "Priya Sharma", "priya.sharma@palletdynamics.example", "Engineering", "Senior Software Engineer", "Bengaluru", 1, "2021-07-19"),
    (3, "Rahul Verma", "rahul.verma@palletdynamics.example", "Engineering", "Data Engineer", "Pune", 1, "2023-01-09"),
    (4, "Meera Iyer", "meera.iyer@palletdynamics.example", "Human Resources", "HR Business Partner", "Bengaluru", None, "2019-11-11"),
    (5, "Arjun Nair", "arjun.nair@palletdynamics.example", "Sales", "Account Executive", "Mumbai", 7, "2024-03-04"),
    (6, "Kavya Menon", "kavya.menon@palletdynamics.example", "Finance", "Finance Manager", "Mumbai", None, "2020-06-15"),
    (7, "Vikram Singh", "vikram.singh@palletdynamics.example", "Sales", "Sales Director", "Delhi", None, "2017-09-25"),
    (8, "Sneha Kulkarni", "sneha.kulkarni@palletdynamics.example", "Engineering", "QA Engineer", "Pune", 1, "2025-08-18"),
    (9, "Lukas Becker", "lukas.becker@palletdynamics.example", "Sales", "Partner Manager", "Berlin", 7, "2022-02-01"),
    (10, "Fatima Sheikh", "fatima.sheikh@palletdynamics.example", "IT", "IT Support Lead", "Bengaluru", None, "2020-01-20"),
]

LEAVE_BALANCES = [
    # employee_id, year, annual_total, annual_used, sick_total, sick_used
    (1, 2026, 32, 18, 12, 2), (2, 2026, 30, 11, 12, 4), (3, 2026, 24, 20, 12, 7),
    (4, 2026, 28, 9, 12, 0), (5, 2026, 24, 3, 12, 1), (6, 2026, 32, 25, 12, 5),
    (7, 2026, 32, 14, 12, 3), (8, 2026, 6, 1, 3, 0), (9, 2026, 27, 15, 12, 2),
    (10, 2026, 26, 22, 12, 9),
]

TICKETS = [
    (101, 2, "Laptop battery swelling", "hardware", "P2", "in_progress", "2026-09-22", None),
    (102, 3, "Need access to analytics warehouse", "access", "P3", "open", "2026-09-28", None),
    (103, 5, "Suspicious email asking for VPN credentials", "security", "P1", "resolved", "2026-09-15", "2026-09-15"),
    (104, 8, "IDE licence not activating", "software", "P3", "resolved", "2026-08-20", "2026-08-22"),
    (105, 2, "MFA token reset", "access", "P2", "resolved", "2026-07-03", "2026-07-03"),
    (106, 9, "VPN disconnects from Berlin office", "software", "P2", "open", "2026-09-30", None),
    (107, 6, "Request for second monitor", "hardware", "P3", "open", "2026-09-25", None),
    (108, 10, "Ransomware alert on shared drive", "security", "P1", "in_progress", "2026-10-01", None),
]


def seed_company_db(path: Path, overwrite: bool = False) -> bool:
    """Create and populate the demo DB. Returns True if it was (re)created."""
    if path.exists() and not overwrite:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    with sqlite3.connect(path) as db:
        db.executescript(SCHEMA)
        db.executemany("INSERT INTO employees VALUES (?, ?, ?, ?, ?, ?, ?, ?)", EMPLOYEES)
        db.executemany("INSERT INTO leave_balances VALUES (?, ?, ?, ?, ?, ?)", LEAVE_BALANCES)
        db.executemany("INSERT INTO it_tickets VALUES (?, ?, ?, ?, ?, ?, ?, ?)", TICKETS)
    return True
