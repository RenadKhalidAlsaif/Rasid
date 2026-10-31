"""Streamlit front end for the loan agent. Run with: streamlit run app.py"""
import tempfile
import time
from pathlib import Path

import pandas as pd
import streamlit as st

import loan_core as core

st.set_page_config(page_title="Loan Agent", layout="wide")

BASELINE_BEFORE_AI = 60.0  # clear rate before the AI
REFERENCE_RATE = 70.0      # reference used by the monitor
EXTERNAL_REDUCTION_PCT = 20.0  # Fuster et al. (2018): automation ~20% faster (US mortgages)


def decision_icon(decision):
    return {"clear": "🟢", "reject": "🔴", "resubmit": "🔵", "uncertain": "🟡"}.get(decision, "⚪")


PAGES = [
    ("فحص شهادة", "🔍"),
    ("لوحة البيانات", "📊"),
    ("تعديلات الذكاء", "🧠"),
    ("مقارنة الزمن", "⏱️"),
]

if "page" not in st.session_state:
    st.session_state["page"] = PAGES[0][0]

st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] .stButton > button {
        justify-content: flex-start;
        text-align: right;
        direction: rtl;
        border-radius: 14px;
        padding: 0.7rem 1rem;
        margin-bottom: 0.45rem;
        font-size: 1rem;
        font-weight: 600;
        border: 1px solid transparent;
        background: transparent;
        transition: all 0.15s ease;
    }
    section[data-testid="stSidebar"] .stButton > button p {
        text-align: right;
        width: 100%;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        border-color: rgba(99, 102, 241, 0.45);
        background: rgba(99, 102, 241, 0.12);
    }
    section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        color: #ffffff;
        border: none;
        box-shadow: 0 6px 18px rgba(99, 102, 241, 0.35);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 🏦 Loan Agent")
    st.caption("فحص واتخاذ القرار")
    st.divider()
    for name, icon in PAGES:
        active = st.session_state["page"] == name
        if st.button(f"{icon}   {name}", key=f"nav_{name}", use_container_width=True,
                     type="primary" if active else "secondary"):
            st.session_state["page"] = name
            st.rerun()
    st.divider()
    st.caption("نسخة تجريبية")

page = st.session_state["page"]

if page == "فحص شهادة":
    st.header("فحص شهادة الراتب")
    uploaded = st.file_uploader("ارفع صورة الشهادة", type=["png", "jpg", "jpeg"])
    loan_type = st.selectbox("نوع القرض", core.LOAN_TYPES)
    loan_amount = st.number_input("مبلغ القرض المطلوب (ريال)", min_value=0.0,
                                  step=500.0, value=0.0)

    if uploaded is not None:
        st.image(uploaded, caption=uploaded.name, width=400)

    if uploaded is not None and st.button("افحص الشهادة"):
        if loan_amount <= 0:
            st.error("أدخل مبلغ القرض قبل الفحص.")
            st.stop()

        started = time.perf_counter()

        suffix = Path(uploaded.name).suffix or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name

        with st.spinner("جاري القراءة والفحص..."):
            payroll = core.load_payroll()
            sha = core.file_sha256(tmp_path)
            fields, raw_text = core.extract_fields(tmp_path)
            fields["loan_amount"] = loan_amount  # مُدخل من الموظف، مو من الـ OCR
            decision, reason = core.decide(fields, payroll)
            core.add_shadow_dbr(fields, payroll)
            flags = core.security_flags(fields, sha)
            repeat = core.check_repeat_after_approval(fields.get("national_id"))
            if repeat:
                flags.append(repeat)
            if decision == "clear" and flags:
                decision, reason = "uncertain", "; ".join(flags)
            fields["security_flags"] = flags
            duration = round(time.perf_counter() - started, 2)
            core.log_decision(uploaded.name, loan_type, decision, reason, fields,
                              duration_sec=duration)
            if fields.get("national_id"):
                core.register_image(sha, fields["national_id"])

        st.session_state["last"] = {
            "name": uploaded.name, "loan_type": loan_type, "decision": decision,
            "reason": reason, "fields": fields, "raw": raw_text, "flags": flags,
            "duration": duration,
        }

    last = st.session_state.get("last")
    if last:
        st.subheader(f"{decision_icon(last['decision'])} {last['decision'].upper()}")
        st.write(last["reason"])
        if last["flags"]:
            st.warning("تنبيهات الاحتيال: " + " | ".join(last["flags"]))
        engine = core.get_ocr()
        st.write(f"محرك القراءة: **{engine.last_engine or engine.backend}**")
        st.write(f"نوع القرض (اختيارك): **{last['loan_type']}**")
        st.write(f"زمن المعالجة: **{last['duration']} ثانية**")
        st.json(last["fields"])

        if last["decision"] == "resubmit":
            st.info("رسالة جاهزة للعميل، انسخها وأرسلها:")
            st.text_area("نص الرسالة",
                         value=f"مرحباً، لم نتمكن من إكمال طلبك. {last['reason']}",
                         height=100)

        if last["decision"] == "uncertain":
            with st.expander("النص الخام من الـ OCR"):
                st.text(last["raw"])
            with st.form("human_review"):
                st.write("قرار المراجع البشري")
                human = st.radio("القرار", ["clear", "reject"], horizontal=True)
                note = st.text_input("ملاحظة (اختياري)")
                if st.form_submit_button("سجّل قرار المراجع"):
                    core.record_review(last["name"], last["decision"], human, note)
                    st.success("تم تسجيل قرار المراجع.")

elif page == "لوحة البيانات":
    st.header("لوحة البيانات")
    source = st.radio("مصدر البيانات", ["الحقيقية", "تجريبية (وهمية)"], horizontal=True)
    if source == "تجريبية (وهمية)":
        st.error("بيانات تجريبية وهمية، ليست قرارات حقيقية")
        df = core.load_decisions_df(core.DEMO_FILE)
    else:
        df = core.load_decisions_df()

    if df.empty:
        st.info("ما فيه قرارات مسجلة بعد.")
    else:
        types = ["الكل"] + sorted(df["loan_type"].dropna().unique().tolist())
        chosen_type = st.selectbox("فلترة حسب نوع القرض", types)
        view = df if chosen_type == "الكل" else df[df["loan_type"] == chosen_type]

        total = len(view)
        clear_n = int((view["decision"] == "clear").sum())
        clear_rate = round(clear_n / total * 100, 1) if total else 0.0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("إجمالي القرارات", total)
        c2.metric("نسبة القبول الحالية", f"{clear_rate}%")
        c3.metric("قبل الذكاء الاصطناعي", f"{BASELINE_BEFORE_AI}%")
        c4.metric("المرجع الحالي للمونيتور", f"{REFERENCE_RATE}%")

        st.subheader("نسبة القبول شهرياً مقارنة بالوضع قبل الذكاء الاصطناعي")
        monthly = (
            view.groupby("month")
            .agg(total=("decision", "size"),
                 clear=("decision", lambda s: int((s == "clear").sum())),
                 reject=("decision", lambda s: int((s == "reject").sum())),
                 resubmit=("decision", lambda s: int((s == "resubmit").sum())),
                 uncertain=("decision", lambda s: int((s == "uncertain").sum())))
            .reset_index()
        )
        monthly["clear_rate"] = (monthly["clear"] / monthly["total"] * 100).round(1)
        monthly["قبل الذكاء"] = BASELINE_BEFORE_AI
        st.line_chart(monthly.set_index("month")[["clear_rate", "قبل الذكاء"]])
        st.dataframe(monthly, use_container_width=True)

        left, right = st.columns(2)
        with left:
            st.subheader("توزيع القرارات")
            st.bar_chart(view["decision"].value_counts())
        with right:
            st.subheader("القرارات حسب نوع القرض")
            st.bar_chart(view.groupby(["loan_type", "decision"]).size().unstack(fill_value=0))

        st.subheader("آخر القرارات")
        cols = ["timestamp", "form", "loan_type", "decision", "reason", "ocr_engine"]
        st.dataframe(view[cols].sort_values("timestamp", ascending=False), use_container_width=True)

    st.subheader("قرارات المراجعين للحالات المعلقة")
    reviews = core.read_jsonl(core.REVIEWS_FILE)
    if reviews:
        rv = pd.DataFrame(reviews)
        st.bar_chart(rv["human_decision"].value_counts())
        st.dataframe(rv, use_container_width=True)
    else:
        st.info("لا يوجد قرارات مراجعة بعد.")

elif page == "تعديلات الذكاء":
    st.header("تعديلات الذكاء الاصطناعي")
    st.write("القيم الحالية:", core.load_thresholds())

    adjustments = core.read_jsonl(core.ADJUSTMENTS_FILE)
    if not adjustments:
        st.info("ما فيه تعديلات مسجلة.")
    for i, adj in enumerate(adjustments):
        with st.container(border=True):
            st.markdown(f"**{adj.get('threshold')}**: {adj.get('old_value')} → {adj.get('new_value')}")
            st.write(f"السبب: الفرق في نسبة القبول {adj.get('shift_points')} نقطة "
                     f"(وقت التشغيل: {adj.get('run_at')})")
            st.caption(f"مصدر المرجع: {adj.get('baseline_source')}")
            if st.button("تراجع عن هذا التعديل", key=f"revert_{i}"):
                core.revert_adjustment(i)
                st.success("تم الرجوع للقيمة القديمة وتسجيل التراجع.")
                st.rerun()

    st.subheader("حالات رُفعت لمراجعة بشرية")
    escalations = core.read_jsonl(core.ESCALATIONS_FILE)
    if escalations:
        st.dataframe(pd.DataFrame(escalations)[["run_at", "reason"]], use_container_width=True)
    else:
        st.info("لا يوجد.")

elif page == "مقارنة الزمن":
    st.header("زمن الموافقة: النظام مقابل الطريقة اليدوية")

    df = core.load_decisions_df()
    if df.empty or "duration_sec" not in df.columns:
        timed = pd.DataFrame()
    else:
        timed = (df.assign(duration_sec=pd.to_numeric(df["duration_sec"], errors="coerce"))
                   .dropna(subset=["duration_sec"]))

    if timed.empty:
        st.info("ما فيه قياسات زمن بعد. افحص فورم من صفحة الفحص وبيظهر هنا.")
    else:
        median_sec = float(timed["duration_sec"].median())

        c1, c2, c3 = st.columns(3)
        c1.metric("الوسيط (ثانية)", round(median_sec, 2))
        c2.metric("المتوسط (ثانية)", round(timed["duration_sec"].mean(), 2))
        c3.metric("عدد القياسات", len(timed))

        st.subheader("الزمن حسب القرار (وسيط بالثانية)")
        st.bar_chart(timed.groupby("decision")["duration_sec"].median())

        st.subheader("مرجع خارجي")
        st.markdown(
            "دراسة Fuster وزملاؤه (2018): المؤسسات المعتمدة على الأتمتة أسرع في معالجة "
            "الرهن العقاري بحوالي 7 إلى 10 أيام، أي نحو 20%. البيانات أمريكية (HMDA، 2010 إلى 2016)، "
            "وتخص الرهن العقاري وليس القروض الشخصية."
        )
        st.caption("المصدر: FDIC - https://www.fdic.gov/media/168591")

        st.subheader("مقارنة بالطريقة اليدوية الحالية")
        manual_hours = st.number_input(
            "متوسط زمن الموافقة اليدوي (بالساعات، من بيانات البنك)", min_value=0.0, value=0.0
        )

        if manual_hours > 0:
            manual_sec = manual_hours * 3600
            ours_pct = round(max(0.0, (1 - median_sec / manual_sec) * 100), 2)
            st.write(f"النظام أسرع بحوالي **{manual_sec / median_sec:,.0f}** مرة في وقت المعالجة.")

            st.subheader("رسم المقارنة: المرجع الخارجي مقابل البرنامج")
            chart = pd.DataFrame(
                {"نسبة التخفيض في الزمن (%)": [EXTERNAL_REDUCTION_PCT, ours_pct]},
                index=["المرجع الخارجي (Fuster 2018)", "البرنامج (مقاس)"],
            )
            st.bar_chart(chart)
            st.caption(
                "تنبيه: المرجع يقيس تخفيض زمن إغلاق الرهن العقاري في أمريكا، وبرنامجنا يقيس "
                "زمن المعالجة الآلية مقابل زمن يدوي تدخله بنفسك. الفرق في الحجم كبير لأن الزمن اليدوي "
                "يشمل الانتظار والتنقل بين الأقسام، فالنسبتين مو متطابقتين تماماً."
            )
        else:
            st.info("أدخل متوسط الزمن اليدوي لرسم المقارنة.")
            st.caption(
                "الزمن اليدوي لازم يُقاس بنفس التعريف (من التقديم حتى القرار). "
                "وزمن النظام لا يشمل انتظار المراجعة البشرية."
            )