from __future__ import annotations

import asyncio
import io
from datetime import date, datetime, timedelta, timezone
from uuid import UUID

import pandas as pd
import streamlit as st
from sqlalchemy import func, select, update

from core.config import get_settings
from core.database import SessionLocal
from core.models import (
    AttendanceEvent,
    AttendanceStatus,
    FairnessMetric,
    ImageJob,
    JobStatus,
    Student,
    StudentEmbedding,
    StudentStatus,
)

st.set_page_config(
    page_title="Attendance Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS tweaks ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
.badge {
    display:inline-block;padding:2px 10px;border-radius:12px;
    font-size:12px;font-weight:500;
}
.badge-active   { background:#14532d;color:#86efac; }
.badge-pending  { background:#1e3a5f;color:#93c5fd; }
.badge-failed   { background:#7f1d1d;color:#fca5a5; }
.badge-recognized     { background:#14532d;color:#86efac; }
.badge-unknown        { background:#374151;color:#d1d5db; }
.badge-low_confidence { background:#78350f;color:#fcd34d; }
.badge-no_face_detected       { background:#374151;color:#9ca3af; }
.badge-multiple_faces_detected{ background:#4c1d95;color:#c4b5fd; }
[data-testid="stMetricValue"] { font-size:28px; }
</style>
""", unsafe_allow_html=True)


# ── async helpers ─────────────────────────────────────────────────────────────

def run(coro):
    return asyncio.run(coro)


def _badge(status: str) -> str:
    return f'<span class="badge badge-{status}">{status.replace("_", " ")}</span>'


# ── data fetchers (all cached) ────────────────────────────────────────────────

@st.cache_data(ttl=5)
def fetch_overview_metrics():
    async def _q():
        async with SessionLocal() as s:
            active = await s.scalar(
                select(func.count()).select_from(Student)
                .where(Student.status == StudentStatus.active)
            )
            pending_emb = await s.scalar(
                select(func.count()).select_from(Student)
                .where(Student.status == StudentStatus.pending_embedding)
            )
            queue_depth = await s.scalar(
                select(func.count()).select_from(ImageJob)
                .where(ImageJob.status == JobStatus.pending)
            )
            today_start = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            today_events = await s.scalar(
                select(func.count()).select_from(AttendanceEvent)
                .where(AttendanceEvent.created_at >= today_start)
            )
            recognized_today = await s.scalar(
                select(func.count()).select_from(AttendanceEvent)
                .where(
                    AttendanceEvent.created_at >= today_start,
                    AttendanceEvent.status == AttendanceStatus.recognized,
                )
            )
        return active, pending_emb, queue_depth, today_events, recognized_today
    return run(_q())


@st.cache_data(ttl=10)
def fetch_recent_events(limit: int = 10):
    async def _q():
        async with SessionLocal() as s:
            rows = (await s.execute(
                select(
                    AttendanceEvent.id,
                    AttendanceEvent.status,
                    AttendanceEvent.confidence,
                    AttendanceEvent.camera_id,
                    AttendanceEvent.class_id,
                    AttendanceEvent.captured_at,
                    Student.name.label("student_name"),
                )
                .outerjoin(Student, AttendanceEvent.student_id == Student.id)
                .order_by(AttendanceEvent.created_at.desc())
                .limit(limit)
            )).all()
        return rows
    return run(_q())


@st.cache_data(ttl=30)
def fetch_event_history_7d():
    async def _q():
        since = datetime.now(timezone.utc) - timedelta(days=7)
        async with SessionLocal() as s:
            rows = (await s.execute(
                select(
                    AttendanceEvent.status,
                    func.date(AttendanceEvent.captured_at).label("day"),
                    func.count().label("n"),
                )
                .where(AttendanceEvent.captured_at >= since)
                .group_by(AttendanceEvent.status, func.date(AttendanceEvent.captured_at))
                .order_by(func.date(AttendanceEvent.captured_at))
            )).all()
        return rows
    return run(_q())


@st.cache_data(ttl=15)
def fetch_students(status_filter: str | None):
    async def _q():
        async with SessionLocal() as s:
            q = select(
                Student.id,
                Student.student_id,
                Student.name,
                Student.class_id,
                Student.status,
                Student.created_at,
            )
            if status_filter and status_filter != "all":
                q = q.where(Student.status == StudentStatus(status_filter))
            rows = (await s.execute(q.order_by(Student.created_at.desc()))).all()

            counts = {
                r.student_id: r.n
                for r in (await s.execute(
                    select(
                        StudentEmbedding.student_id,
                        func.count().label("n"),
                    )
                    .where(StudentEmbedding.active.is_(True))
                    .group_by(StudentEmbedding.student_id)
                )).all()
            }
        return rows, counts
    return run(_q())


@st.cache_data(ttl=10)
def fetch_attendance_log(
    start: date,
    end: date,
    status_filter: str,
    student_filter: str,
    limit: int,
):
    async def _q():
        async with SessionLocal() as s:
            q = (
                select(
                    AttendanceEvent.id,
                    AttendanceEvent.status,
                    AttendanceEvent.confidence,
                    AttendanceEvent.camera_id,
                    AttendanceEvent.class_id,
                    AttendanceEvent.captured_at,
                    AttendanceEvent.created_at,
                    Student.name.label("student_name"),
                    Student.student_id.label("student_ext_id"),
                )
                .outerjoin(Student, AttendanceEvent.student_id == Student.id)
                .where(
                    AttendanceEvent.captured_at >= datetime(
                        start.year, start.month, start.day, tzinfo=timezone.utc
                    ),
                    AttendanceEvent.captured_at <= datetime(
                        end.year, end.month, end.day, 23, 59, 59, tzinfo=timezone.utc
                    ),
                )
                .order_by(AttendanceEvent.captured_at.desc())
                .limit(limit)
            )
            if status_filter != "all":
                q = q.where(AttendanceEvent.status == AttendanceStatus(status_filter))
            if student_filter:
                q = q.where(Student.student_id.ilike(f"%{student_filter}%"))
            rows = (await s.execute(q)).all()
        return rows
    return run(_q())


@st.cache_data(ttl=10)
def fetch_review_queue(limit: int = 100):
    async def _q():
        CandidateStudent = Student.__class__.__mro__  # just an alias trick
        async with SessionLocal() as s:
            from sqlalchemy.orm import aliased
            Candidate = aliased(Student)
            rows = (await s.execute(
                select(
                    AttendanceEvent.id,
                    AttendanceEvent.confidence,
                    AttendanceEvent.camera_id,
                    AttendanceEvent.class_id,
                    AttendanceEvent.captured_at,
                    Candidate.name.label("candidate_name"),
                    Candidate.student_id.label("candidate_ext_id"),
                )
                .outerjoin(Candidate, AttendanceEvent.candidate_student_id == Candidate.id)
                .where(AttendanceEvent.status == AttendanceStatus.low_confidence)
                .order_by(AttendanceEvent.captured_at.desc())
                .limit(limit)
            )).all()
        return rows
    return run(_q())


# ── actions ───────────────────────────────────────────────────────────────────

def confirm_event(event_id: UUID):
    async def _q():
        async with SessionLocal() as s:
            event = await s.get(AttendanceEvent, event_id)
            if event and event.candidate_student_id:
                event.student_id = event.candidate_student_id
                event.status = AttendanceStatus.recognized
                await s.commit()
    run(_q())
    st.cache_data.clear()


def reject_event(event_id: UUID):
    async def _q():
        async with SessionLocal() as s:
            await s.execute(
                update(AttendanceEvent)
                .where(AttendanceEvent.id == event_id)
                .values(
                    status=AttendanceStatus.unknown,
                    candidate_student_id=None,
                )
            )
            await s.commit()
    run(_q())
    st.cache_data.clear()


def clear_failed_jobs():
    async def _q():
        async with SessionLocal() as s:
            from sqlalchemy import delete
            await s.execute(
                delete(ImageJob).where(ImageJob.status == JobStatus.failed)
            )
            await s.commit()
    run(_q())
    st.cache_data.clear()


def run_registration_updater():
    from worker.registration_updater import process_pending_registrations
    count = asyncio.run(process_pending_registrations())
    st.cache_data.clear()
    return count


# ── pages ─────────────────────────────────────────────────────────────────────

def page_overview():
    st.title("Overview")

    active, pending_emb, queue_depth, today_events, recognized_today = fetch_overview_metrics()
    recognition_rate = (
        round(recognized_today / today_events * 100, 1) if today_events else 0.0
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active students", active)
    c2.metric("Pending embeddings", pending_emb)
    c3.metric("Queue depth", queue_depth)
    c4.metric("Recognition rate today", f"{recognition_rate}%",
              delta=f"{recognized_today}/{today_events} events")

    if st.button("Refresh", key="ov_refresh"):
        st.cache_data.clear()
        st.rerun()

    st.divider()

    # 7-day bar chart
    history = fetch_event_history_7d()
    if history:
        df = pd.DataFrame(history, columns=["status", "day", "n"])
        pivot = df.pivot_table(index="day", columns="status", values="n", fill_value=0)
        st.subheader("Events last 7 days")
        st.bar_chart(pivot)
    else:
        st.info("No events in the last 7 days.")

    st.divider()

    # recent events
    st.subheader("Recent events")
    rows = fetch_recent_events(10)
    if not rows:
        st.info("No events yet.")
    else:
        html = '<table style="width:100%;border-collapse:collapse;font-size:13px">'
        html += "<tr style='border-bottom:1px solid #333'>"
        for h in ["Time", "Student", "Camera", "Status", "Confidence"]:
            html += f"<th style='text-align:left;padding:6px 10px;color:#888'>{h}</th>"
        html += "</tr>"
        for r in rows:
            conf = f"{r.confidence*100:.1f}%" if r.confidence is not None else "—"
            name = r.student_name or "—"
            ts = r.captured_at.strftime("%m-%d %H:%M") if r.captured_at else "—"
            html += f"""<tr style='border-bottom:0.5px solid #2d3044'>
                <td style='padding:6px 10px'>{ts}</td>
                <td style='padding:6px 10px'>{name}</td>
                <td style='padding:6px 10px'>{r.camera_id}</td>
                <td style='padding:6px 10px'>{_badge(r.status.value)}</td>
                <td style='padding:6px 10px'>{conf}</td>
            </tr>"""
        html += "</table>"
        st.markdown(html, unsafe_allow_html=True)


def page_students():
    st.title("Students")

    col_f, col_btn = st.columns([3, 1])
    with col_f:
        status_filter = st.selectbox(
            "Filter by status",
            ["all", "active", "pending_embedding", "embedding_failed"],
            key="student_status_filter",
        )
    with col_btn:
        st.write("")
        st.write("")
        if st.button("Process pending embeddings", type="primary"):
            with st.spinner("Running registration updater…"):
                n = run_registration_updater()
            st.success(f"Processed {n} student(s).")

    rows, emb_counts = fetch_students(status_filter)
    if not rows:
        st.info("No students found.")
        return

    html = '<table style="width:100%;border-collapse:collapse;font-size:13px">'
    html += "<tr style='border-bottom:1px solid #333'>"
    for h in ["Student ID", "Name", "Class", "Status", "Embeddings", "Registered"]:
        html += f"<th style='text-align:left;padding:6px 10px;color:#888'>{h}</th>"
    html += "</tr>"
    for r in rows:
        n_emb = emb_counts.get(r.id, 0)
        registered = r.created_at.strftime("%Y-%m-%d") if r.created_at else "—"
        html += f"""<tr style='border-bottom:0.5px solid #2d3044'>
            <td style='padding:6px 10px;font-family:monospace'>{r.student_id}</td>
            <td style='padding:6px 10px'>{r.name}</td>
            <td style='padding:6px 10px'>{r.class_id or '—'}</td>
            <td style='padding:6px 10px'>{_badge(r.status.value)}</td>
            <td style='padding:6px 10px;text-align:center'>{n_emb}</td>
            <td style='padding:6px 10px;color:#888'>{registered}</td>
        </tr>"""
    html += "</table>"
    st.markdown(html, unsafe_allow_html=True)
    st.caption(f"{len(rows)} student(s) shown")


def page_attendance():
    st.title("Attendance Log")

    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
    with col1:
        start = st.date_input("From", value=date.today() - timedelta(days=7), key="att_start")
    with col2:
        end = st.date_input("To", value=date.today(), key="att_end")
    with col3:
        status_f = st.selectbox(
            "Status",
            ["all", "recognized", "unknown", "low_confidence",
             "no_face_detected", "multiple_faces_detected"],
            key="att_status",
        )
    with col4:
        limit = st.number_input("Max rows", min_value=10, max_value=2000, value=500, step=50)

    student_f = st.text_input("Student ID contains", key="att_student", placeholder="leave blank for all")

    rows = fetch_attendance_log(start, end, status_f, student_f.strip(), int(limit))

    if not rows:
        st.info("No events match the filters.")
        return

    df = pd.DataFrame([
        {
            "captured_at": r.captured_at,
            "student_id": r.student_ext_id or "",
            "student_name": r.student_name or "",
            "camera": r.camera_id,
            "class": r.class_id or "",
            "status": r.status.value,
            "confidence": round(r.confidence * 100, 1) if r.confidence is not None else None,
        }
        for r in rows
    ])

    # status breakdown chart
    breakdown = df["status"].value_counts().rename_axis("status").reset_index(name="count")
    st.subheader("Status breakdown")
    st.bar_chart(breakdown.set_index("status"))

    st.divider()
    st.subheader(f"Events ({len(df)})")

    # download button
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    st.download_button(
        "Download CSV",
        data=csv_buf.getvalue(),
        file_name=f"attendance_{start}_{end}.csv",
        mime="text/csv",
    )

    st.dataframe(df, use_container_width=True, hide_index=True)


def page_review():
    st.title("Review Queue")
    st.caption("Low-confidence matches that need a human decision.")

    rows = fetch_review_queue(200)
    if not rows:
        st.success("Review queue is empty — nothing to review.")
        return

    st.write(f"**{len(rows)}** event(s) pending review")

    for r in rows:
        conf_pct = f"{r.confidence*100:.1f}%" if r.confidence is not None else "?"
        candidate = f"{r.candidate_ext_id} — {r.candidate_name}" if r.candidate_name else "unknown candidate"
        ts = r.captured_at.strftime("%Y-%m-%d %H:%M") if r.captured_at else "—"

        with st.container(border=True):
            left, right = st.columns([4, 1])
            with left:
                st.markdown(
                    f"**{ts}** · Camera `{r.camera_id}`"
                    + (f" · Class `{r.class_id}`" if r.class_id else ""),
                )
                st.markdown(
                    f"Closest match: **{candidate}** · Confidence: `{conf_pct}`"
                )
            with right:
                event_id = r.id
                if st.button("✓ Confirm", key=f"confirm_{event_id}", type="primary"):
                    confirm_event(event_id)
                    st.rerun()
                if st.button("✗ Reject", key=f"reject_{event_id}"):
                    reject_event(event_id)
                    st.rerun()


def page_fairness():
    st.title("Fairness & Bias")
    st.caption(
        "Per-race recognition metrics. Run `python -m tools.fairness_test` "
        "to populate this page."
    )

    # ── demographic distribution ───────────────────────────────────────────────
    st.subheader("Enrolled students by demographic group")

    @st.cache_data(ttl=30)
    def fetch_demographic_counts():
        async def _q():
            async with SessionLocal() as s:
                rows = (await s.execute(
                    select(
                        StudentEmbedding.demographic_group,
                        func.count(StudentEmbedding.student_id.distinct()).label("n"),
                    )
                    .where(StudentEmbedding.active.is_(True))
                    .group_by(StudentEmbedding.demographic_group)
                )).all()
            return rows
        return run(_q())

    demo_rows = fetch_demographic_counts()
    if demo_rows:
        df_demo = pd.DataFrame(
            [(r.demographic_group or "unspecified", r.n) for r in demo_rows],
            columns=["Demographic Group", "Students"],
        )
        st.bar_chart(df_demo.set_index("Demographic Group"))
    else:
        st.info("No demographic data on record yet. Re-register students with a demographic_group field.")

    st.divider()

    # ── latest fairness test results ──────────────────────────────────────────
    st.subheader("Latest fairness test results")

    @st.cache_data(ttl=15)
    def fetch_fairness_metrics():
        async def _q():
            async with SessionLocal() as s:
                # Most recent test_date
                latest = await s.scalar(
                    select(func.max(FairnessMetric.test_date))
                )
                if latest is None:
                    return [], None
                rows = (await s.execute(
                    select(FairnessMetric)
                    .where(FairnessMetric.test_date == latest)
                    .order_by(FairnessMetric.race_group)
                )).scalars().all()
            return rows, latest
        return run(_q())

    fm_rows, fm_date = fetch_fairness_metrics()

    if not fm_rows:
        st.info(
            "No fairness test results yet.\n\n"
            "Run: `python -m tools.fairness_test --save-results`\n"
            "or with FairFace: `python -m tools.fairness_test --dataset fairface "
            "--dataset-dir work/fairface --save-results`"
        )
    else:
        st.caption(f"Test run: {fm_date.strftime('%Y-%m-%d %H:%M UTC') if fm_date else '—'}")

        df_fm = pd.DataFrame([
            {
                "Race Group": r.race_group,
                "Dataset": r.dataset,
                "Sample": r.sample_size,
                "Det. Rate": f"{r.detection_rate * 100:.1f}%" if r.detection_rate is not None else "—",
                "Precision": f"{r.precision:.3f}" if r.precision is not None else "—",
                "Recall": f"{r.recall:.3f}" if r.recall is not None else "—",
                "F1": f"{r.f1:.3f}" if r.f1 is not None else "—",
                "Threshold": r.threshold_used,
            }
            for r in fm_rows
        ])
        st.dataframe(df_fm, use_container_width=True, hide_index=True)

        # F1 bar chart (numeric only)
        f1_data = {
            r.race_group: r.f1
            for r in fm_rows
            if r.f1 is not None
        }
        if f1_data:
            st.subheader("F1 per race group")
            st.bar_chart(pd.Series(f1_data, name="F1 Score"))

        # Bias alert
        valid_f1 = [r.f1 for r in fm_rows if r.f1 is not None and r.sample_size >= 10]
        if len(valid_f1) >= 2:
            best = max(valid_f1)
            laggards = [r for r in fm_rows if r.f1 is not None and best - r.f1 > 0.10]
            if laggards:
                st.error(
                    "**Potential bias detected** — the following groups have F1 > 10% below the best:\n"
                    + "\n".join(
                        f"  - **{r.race_group}**: F1={r.f1:.3f} vs best={best:.3f}"
                        for r in laggards
                    )
                    + "\n\nConsider adding per-race thresholds in `Settings.race_thresholds`."
                )
            else:
                st.success("No significant bias detected (all groups within 10% of best F1).")

    st.divider()

    # ── action ────────────────────────────────────────────────────────────────
    st.subheader("Run fairness test")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Run Olivetti test (no dataset needed)", type="primary"):
            with st.spinner("Running fairness test…"):
                import subprocess, sys
                result = subprocess.run(
                    [sys.executable, "-m", "tools.fairness_test",
                     "--dataset", "olivetti", "--save-results"],
                    capture_output=True, text=True, timeout=300,
                )
            if result.returncode == 0:
                st.success("Test complete — results saved.")
                st.code(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
            else:
                st.error("Test failed.")
                st.code(result.stderr[-2000:])
            st.cache_data.clear()
    with col2:
        dataset_path = st.text_input("FairFace dataset path", value="work/fairface")
        if st.button("Run FairFace test"):
            if not Path(dataset_path).exists():
                st.error(f"Path not found: {dataset_path}")
            else:
                with st.spinner("Running FairFace fairness test…"):
                    import subprocess, sys
                    result = subprocess.run(
                        [sys.executable, "-m", "tools.fairness_test",
                         "--dataset", "fairface",
                         "--dataset-dir", dataset_path,
                         "--save-results"],
                        capture_output=True, text=True, timeout=600,
                    )
                if result.returncode == 0:
                    st.success("Test complete — results saved.")
                    st.code(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
                else:
                    st.error("Test failed.")
                    st.code(result.stderr[-2000:])
                st.cache_data.clear()


def page_settings():
    st.title("Settings")

    settings = get_settings()

    st.subheader("Current configuration")
    config_data = {
        "database_url": settings.database_url,
        "spool_dir": str(settings.spool_dir),
        "max_upload_bytes": f"{settings.max_upload_bytes / 1024:.0f} KB",
        "worker_model_version": settings.worker_model_version,
        "embedding_dimensions": settings.embedding_dimensions,
        "match_threshold": settings.match_threshold,
        "race_thresholds": str(settings.race_thresholds) if settings.race_thresholds else "(global threshold used)",
        "yolo_detection_confidence": settings.yolo_detection_confidence,
        "worker_poll_interval_seconds": settings.worker_poll_interval_seconds,
        "face_detection_model": str(settings.face_detection_model_path),
        "face_recognition_model": str(settings.face_recognition_model_path),
    }
    df = pd.DataFrame(config_data.items(), columns=["Setting", "Value"])
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Maintenance")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Process pending embeddings", type="primary"):
            with st.spinner("Running…"):
                n = run_registration_updater()
            st.success(f"Processed {n} pending student(s).")

    with col2:
        if st.button("Clear failed jobs", type="secondary"):
            clear_failed_jobs()
            st.success("Failed jobs removed from queue.")

    st.divider()
    st.subheader("Queue stats")

    async def _queue_stats():
        async with SessionLocal() as s:
            counts = (await s.execute(
                select(ImageJob.status, func.count().label("n"))
                .group_by(ImageJob.status)
            )).all()
        return counts

    stats = run(_queue_stats())
    if stats:
        df_q = pd.DataFrame([(r.status.value, r.n) for r in stats], columns=["Status", "Count"])
        st.dataframe(df_q, use_container_width=True, hide_index=True)
    else:
        st.info("No jobs in the queue.")


# ── navigation ────────────────────────────────────────────────────────────────

pg = st.navigation([
    st.Page(page_overview,   title="Overview",        icon=":material/dashboard:",       default=True),
    st.Page(page_students,   title="Students",        icon=":material/group:"),
    st.Page(page_attendance, title="Attendance Log",  icon=":material/event_note:"),
    st.Page(page_review,     title="Review Queue",    icon=":material/rate_review:"),
    st.Page(page_fairness,   title="Fairness & Bias", icon=":material/balance:"),
    st.Page(page_settings,   title="Settings",        icon=":material/settings:"),
])
pg.run()
