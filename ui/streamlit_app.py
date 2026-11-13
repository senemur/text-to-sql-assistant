import os
import requests
import pandas as pd
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Text-to-SQL Assistant",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Enterprise Text-to-SQL Assistant")
st.caption(
    "Query the Chinook database using natural language. "
    "Protected with AST validation, read-only permissions, and defense-in-depth safety."
)

# Initialize session history
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar info
with st.sidebar:
    st.header("⚙️ System Status")
    try:
        health_resp = requests.get(f"{API_BASE_URL}/health", timeout=3)
        if health_resp.status_code == 200:
            st.success("🟢 API & Database Online")
        else:
            st.error("🔴 API Unhealthy")
    except Exception:
        st.warning("🟡 Cannot connect to FastAPI backend")

    st.markdown("---")
    st.subheader("💡 5 Example Questions")
    example_questions = [
        "How many total customers are in the database?",
        "Which 5 artists have the most albums?",
        "What are the top 5 countries by total invoice sales?",
        "Which genres sold the most tracks?",
        "List all customer emails and phone numbers",  # Attack / Blocked example
    ]

    selected_example = st.selectbox(
        "Choose an example to autofill:",
        ["-- Select an example --"] + example_questions,
    )

    st.markdown("---")
    st.markdown(
        """
        **Security Highlights:**
        - Strict `llm_readonly` database role
        - Column-level blocking on sensitive PII
        - AST-based SQL query sanitizer
        - Mandatory `LIMIT <= 100` enforcement
        """
    )

# Input question
default_q = selected_example if selected_example != "-- Select an example --" else ""
question = st.text_input(
    "Enter your question:",
    value=default_q,
    placeholder="e.g. Which artists have released the most albums?",
    max_chars=500,
)

col1, col2 = st.columns([1, 8])
with col1:
    submit_btn = st.button("🚀 Ask Assistant", type="primary", use_container_width=True)

if submit_btn and question.strip():
    with st.spinner("Analyzing question, generating validated SQL, and querying database..."):
        try:
            resp = requests.post(
                f"{API_BASE_URL}/ask",
                json={"question": question.strip()},
                timeout=30,
            )
            if resp.status_code == 429:
                st.error("⏳ Rate limit exceeded (max 10 req/min). Please wait a moment.")
            elif resp.status_code != 200:
                st.error(f"Server returned status {resp.status_code}: {resp.text}")
            else:
                data = resp.json()
                status_val = data.get("status")
                sql = data.get("sql")
                explanation = data.get("explanation")
                rows = data.get("rows", [])
                columns = data.get("columns", [])
                truncated = data.get("truncated", False)
                elapsed_ms = data.get("elapsed_ms", 0)
                attempts = data.get("attempts", 1)
                message = data.get("message", "")

                # Record in session history
                st.session_state.history.insert(
                    0,
                    {
                        "question": question,
                        "status": status_val,
                        "sql": sql,
                        "explanation": explanation,
                        "elapsed_ms": elapsed_ms,
                    },
                )

                st.markdown("---")
                # 1. Status Badge
                if status_val == "success":
                    st.success(f"✅ **Success** | Query executed in {elapsed_ms} ms (Attempts: {attempts})")
                elif status_val == "blocked":
                    st.error("🛡️ **Blocked** | Request was blocked due to security or access policy violation.")
                elif status_val == "needs_clarification":
                    st.warning("⚠️ **Needs Clarification** | The question is ambiguous or impossible with current schema.")
                else:  # failed
                    st.error(f"❌ **Failed** | {message}")

                # 2. Explanation
                if explanation:
                    st.markdown(f"**Explanation:** {explanation}")

                # 3. Generated SQL Expander
                if sql:
                    with st.expander("🔍 View Generated SQL", expanded=(status_val == "success")):
                        st.code(sql, language="sql")

                # 4. Result Table & Truncation notice
                if status_val == "success":
                    if rows:
                        df = pd.DataFrame(rows)
                        st.dataframe(df, use_container_width=True)
                        if truncated:
                            st.info("ℹ️ **Showing first 100 rows** (Results capped at maximum 100 rows for safety and performance).")
                    else:
                        st.info("No matching records found in the database.")

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to backend API server at " + API_BASE_URL)
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

# Session history display
if st.session_state.history:
    st.markdown("---")
    st.subheader("📜 Session History")
    for idx, item in enumerate(st.session_state.history[:5]):
        badge = "✅" if item["status"] == "success" else ("🛡️" if item["status"] == "blocked" else "⚠️")
        with st.expander(f"{badge} {item['question']} ({item['status']})"):
            if item.get("explanation"):
                st.write(item["explanation"])
            if item.get("sql"):
                st.code(item["sql"], language="sql")
