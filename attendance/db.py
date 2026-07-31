"""SQLite storage for the attendance system.

Owns all domain state: users, courses, enrolments, class sessions and
attendance records. Face embeddings live in the separate faceapi service —
here we only keep the `face_subject_id` that links a user to that gallery.
"""
from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
    id              TEXT PRIMARY KEY,
    email           TEXT UNIQUE NOT NULL,
    password_hash   TEXT NOT NULL,
    full_name       TEXT NOT NULL,
    role            TEXT NOT NULL CHECK (role IN ('student','teacher','admin')),
    student_number  TEXT UNIQUE,
    face_subject_id TEXT,                 -- subject_id in the faceapi gallery
    face_enrolled   INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS courses (
    id          TEXT PRIMARY KEY,
    code        TEXT UNIQUE NOT NULL,
    name        TEXT NOT NULL,
    teacher_id  TEXT REFERENCES users(id) ON DELETE SET NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS enrolments (
    course_id   TEXT NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    student_id  TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at  TEXT NOT NULL,
    PRIMARY KEY (course_id, student_id)
);

CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    course_id   TEXT NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    title       TEXT,
    status      TEXT NOT NULL CHECK (status IN ('open','closed')) DEFAULT 'open',
    opened_at   TEXT NOT NULL,
    closed_at   TEXT,
    opened_by   TEXT REFERENCES users(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS attendance (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    student_id  TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    marked_at   TEXT NOT NULL,
    method      TEXT NOT NULL CHECK (method IN ('face','manual')),
    confidence  REAL,
    marked_by   TEXT REFERENCES users(id) ON DELETE SET NULL,
    UNIQUE (session_id, student_id)
);

CREATE INDEX IF NOT EXISTS idx_att_session ON attendance(session_id);
CREATE INDEX IF NOT EXISTS idx_att_student ON attendance(student_id);
CREATE INDEX IF NOT EXISTS idx_sessions_course ON sessions(course_id, status);
CREATE INDEX IF NOT EXISTS idx_enrol_student ON enrolments(student_id);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    return uuid.uuid4().hex


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ── generic helpers ──────────────────────────────────────────────────────
    def q(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        return self.conn.execute(sql, params).fetchall()

    def one(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        return self.conn.execute(sql, params).fetchone()

    def run(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        cur = self.conn.execute(sql, params)
        self.conn.commit()
        return cur

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    # ── users ────────────────────────────────────────────────────────────────
    def create_user(self, email, password_hash, full_name, role,
                    student_number=None) -> str:
        uid = new_id()
        self.run(
            "INSERT INTO users(id,email,password_hash,full_name,role,student_number,"
            "face_subject_id,face_enrolled,created_at) VALUES(?,?,?,?,?,?,?,0,?)",
            (uid, email.lower().strip(), password_hash, full_name, role,
             student_number, f"user-{uid}", now_iso()))
        return uid

    def user_by_email(self, email: str):
        return self.one("SELECT * FROM users WHERE email=?", (email.lower().strip(),))

    def user_by_id(self, uid: str):
        return self.one("SELECT * FROM users WHERE id=?", (uid,))

    def user_by_subject(self, subject_id: str):
        return self.one("SELECT * FROM users WHERE face_subject_id=?", (subject_id,))

    def mark_face_enrolled(self, uid: str, enrolled: bool = True) -> None:
        self.run("UPDATE users SET face_enrolled=? WHERE id=?", (1 if enrolled else 0, uid))

    def list_users(self, role: str | None = None):
        if role:
            return self.q("SELECT * FROM users WHERE role=? ORDER BY full_name", (role,))
        return self.q("SELECT * FROM users ORDER BY role, full_name")

    # ── courses & enrolments ─────────────────────────────────────────────────
    def create_course(self, code, name, teacher_id=None) -> str:
        cid = new_id()
        self.run("INSERT INTO courses(id,code,name,teacher_id,created_at) VALUES(?,?,?,?,?)",
                 (cid, code.strip(), name.strip(), teacher_id, now_iso()))
        return cid

    def list_courses(self):
        return self.q(
            "SELECT c.*, u.full_name AS teacher_name, "
            "(SELECT COUNT(*) FROM enrolments e WHERE e.course_id=c.id) AS student_count "
            "FROM courses c LEFT JOIN users u ON u.id=c.teacher_id ORDER BY c.code")

    def courses_for_student(self, student_id: str):
        return self.q(
            "SELECT c.* FROM courses c JOIN enrolments e ON e.course_id=c.id "
            "WHERE e.student_id=? ORDER BY c.code", (student_id,))

    def enrol(self, course_id: str, student_id: str) -> None:
        self.run("INSERT OR IGNORE INTO enrolments(course_id,student_id,created_at) "
                 "VALUES(?,?,?)", (course_id, student_id, now_iso()))

    def unenrol(self, course_id: str, student_id: str) -> None:
        self.run("DELETE FROM enrolments WHERE course_id=? AND student_id=?",
                 (course_id, student_id))

    def course_students(self, course_id: str):
        return self.q(
            "SELECT u.* FROM users u JOIN enrolments e ON e.student_id=u.id "
            "WHERE e.course_id=? ORDER BY u.full_name", (course_id,))

    def is_enrolled(self, course_id: str, student_id: str) -> bool:
        return self.one("SELECT 1 FROM enrolments WHERE course_id=? AND student_id=?",
                        (course_id, student_id)) is not None

    # ── sessions ─────────────────────────────────────────────────────────────
    def open_session(self, course_id: str, title: str | None, opened_by: str) -> str:
        sid = new_id()
        self.run("INSERT INTO sessions(id,course_id,title,status,opened_at,opened_by) "
                 "VALUES(?,?,?,'open',?,?)", (sid, course_id, title, now_iso(), opened_by))
        return sid

    def close_session(self, session_id: str) -> None:
        self.run("UPDATE sessions SET status='closed', closed_at=? WHERE id=?",
                 (now_iso(), session_id))

    def session(self, session_id: str):
        return self.one("SELECT * FROM sessions WHERE id=?", (session_id,))

    def open_session_for_course(self, course_id: str):
        return self.one("SELECT * FROM sessions WHERE course_id=? AND status='open' "
                        "ORDER BY opened_at DESC LIMIT 1", (course_id,))

    def list_sessions(self, course_id: str | None = None):
        base = ("SELECT s.*, c.code AS course_code, c.name AS course_name, "
                "(SELECT COUNT(*) FROM attendance a WHERE a.session_id=s.id) AS present "
                "FROM sessions s JOIN courses c ON c.id=s.course_id ")
        if course_id:
            return self.q(base + "WHERE s.course_id=? ORDER BY s.opened_at DESC", (course_id,))
        return self.q(base + "ORDER BY s.opened_at DESC LIMIT 100")

    # ── attendance ───────────────────────────────────────────────────────────
    def already_marked(self, session_id: str, student_id: str):
        return self.one("SELECT * FROM attendance WHERE session_id=? AND student_id=?",
                        (session_id, student_id))

    def mark_attendance(self, session_id, student_id, method, confidence=None,
                        marked_by=None) -> str:
        aid = new_id()
        self.run("INSERT INTO attendance(id,session_id,student_id,marked_at,method,"
                 "confidence,marked_by) VALUES(?,?,?,?,?,?,?)",
                 (aid, session_id, student_id, now_iso(), method, confidence, marked_by))
        return aid

    def unmark_attendance(self, session_id: str, student_id: str) -> None:
        self.run("DELETE FROM attendance WHERE session_id=? AND student_id=?",
                 (session_id, student_id))

    def session_attendance(self, session_id: str):
        return self.q(
            "SELECT a.*, u.full_name, u.student_number FROM attendance a "
            "JOIN users u ON u.id=a.student_id WHERE a.session_id=? "
            "ORDER BY a.marked_at DESC", (session_id,))

    def student_history(self, student_id: str):
        return self.q(
            "SELECT a.marked_at, a.method, a.confidence, s.id AS session_id, s.title, "
            "c.code AS course_code, c.name AS course_name FROM attendance a "
            "JOIN sessions s ON s.id=a.session_id JOIN courses c ON c.id=s.course_id "
            "WHERE a.student_id=? ORDER BY a.marked_at DESC", (student_id,))

    def student_percentages(self, student_id: str):
        """Per-course attendance %: sessions attended / total sessions held."""
        return self.q(
            "SELECT c.id, c.code, c.name, "
            "(SELECT COUNT(*) FROM sessions s WHERE s.course_id=c.id) AS total, "
            "(SELECT COUNT(*) FROM attendance a JOIN sessions s2 ON s2.id=a.session_id "
            "  WHERE s2.course_id=c.id AND a.student_id=?) AS attended "
            "FROM courses c JOIN enrolments e ON e.course_id=c.id "
            "WHERE e.student_id=? ORDER BY c.code", (student_id, student_id))

    def course_report(self, course_id: str):
        """Per-student attended/total for one course."""
        return self.q(
            "SELECT u.id, u.full_name, u.student_number, "
            "(SELECT COUNT(*) FROM sessions s WHERE s.course_id=?) AS total, "
            "(SELECT COUNT(*) FROM attendance a JOIN sessions s2 ON s2.id=a.session_id "
            "  WHERE s2.course_id=? AND a.student_id=u.id) AS attended "
            "FROM users u JOIN enrolments e ON e.student_id=u.id "
            "WHERE e.course_id=? ORDER BY u.full_name",
            (course_id, course_id, course_id))

    def stats(self) -> dict:
        def c(sql):
            return self.one(sql)[0]
        return {
            "students": c("SELECT COUNT(*) FROM users WHERE role='student'"),
            "teachers": c("SELECT COUNT(*) FROM users WHERE role='teacher'"),
            "courses": c("SELECT COUNT(*) FROM courses"),
            "sessions": c("SELECT COUNT(*) FROM sessions"),
            "open_sessions": c("SELECT COUNT(*) FROM sessions WHERE status='open'"),
            "attendance_records": c("SELECT COUNT(*) FROM attendance"),
            "face_enrolled": c("SELECT COUNT(*) FROM users WHERE face_enrolled=1"),
        }
