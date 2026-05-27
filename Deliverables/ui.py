# app_student.py
import streamlit as st
import pandas as pd
from auth_utils import login_user
from db_utils import get_db_conn

st.set_page_config(page_title="SAMS - Student", layout="wide")

if "user" not in st.session_state:
    st.session_state.user = None

st.title("🎓 Student Portal — Institutional Login")

if not st.session_state.user:
    with st.form("student_login"):
        email = st.text_input("Institutional Email (e.g. name@college.edu)")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

    if submit:
        ok, result = login_user(email, password)
        if ok and result["role"] == "student":
            st.session_state.user = result
            st.success("✅ Logged in successfully")
            st.rerun()
        else:
            st.error(result)
else:
    st.subheader(f"Welcome, {st.session_state.user['email']}")
    conn = get_db_conn()
    cur = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT date, subject, status FROM attendance
        WHERE student_id=%s
        ORDER BY date DESC
    """, (st.session_state.user["id"],))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if rows:
        st.dataframe(pd.DataFrame(rows))
    else:
        st.info("No attendance records yet.")

    if st.button("🚪 Logout"):
        st.session_state.user = None
        st.rerun()
