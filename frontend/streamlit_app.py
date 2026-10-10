import streamlit as st
import requests
import json
import re
from html import escape

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Q&A",
    page_icon="🔍",
    layout="centered",
)

# ── Minimal Custom CSS ───────────────────────────────────────
st.markdown("""
<style>
    .citation-badge {
        display: inline-block;
        background: #f0f2f6;
        border-radius: 6px;
        padding: 4px 10px;
        margin: 2px 4px;
        font-size: 0.82em;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# ── Backend Config ───────────────────────────────────────────
BACKEND_URL = "http://127.0.0.1:8000"

# ── Helper: parse Gemini JSON response ───────────────────────
def parse_answer(raw_text: str):
    """
    The backend prompt (generation.py) asks Gemini to return JSON:
      {"answer": "...", "citations": [{"source": "...", "page": "..."}]}
    But Gemini sometimes wraps it in ```json ... ``` fences.
    This function extracts the clean answer and citations.
    """
    text = (raw_text or "").strip()

    # Strip markdown code fences if present
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data.get("answer", raw_text), data.get("citations", [])
    except (json.JSONDecodeError, TypeError):
        pass

    return raw_text, []




# ── Main UI ──────────────────────────────────────────────────
st.title("🔍 RAG Q&A")
st.caption("Ask a question — answers are grounded in your indexed documents only.")

uploaded_file = st.file_uploader("Document", type=["pdf", "txt"], max_upload_size=25)
upload = st.button("Upload", use_container_width=True, disabled=uploaded_file is None)

if upload and uploaded_file is not None:
    st.session_state.pop("upload_message", None)
    with st.spinner("Preparing your document..."):
        try:
            resp = requests.post(
                f"{BACKEND_URL}/upload",
                params={"filename": uploaded_file.name},
                data=uploaded_file.getvalue(),
                headers={"Content-Type": "application/octet-stream"},
                timeout=(10, 600),
            )
            resp.raise_for_status()
            result = resp.json()
            st.session_state["upload_message"] = f"{result['filename']} is ready for questions."
        except requests.Timeout:
            st.warning("Processing is taking longer than expected. Please wait before trying again.")
        except requests.RequestException as error:
            message = "Upload failed. Check that the document service is running."
            if error.response is not None:
                try:
                    message = error.response.json().get("detail", message)
                except ValueError:
                    pass
            st.error(message)

if "upload_message" in st.session_state:
    st.success(st.session_state["upload_message"])

query = st.text_input(
    "Your question:",
    placeholder="e.g. What products does Nexus Corp offer?",
)

ask = st.button("Ask", type="primary", use_container_width=True)

# ── Query Execution ──────────────────────────────────────────
if ask and query.strip():
    with st.spinner("Running pipeline…"):
        try:
            resp = requests.post(
                f"{BACKEND_URL}/query",
                params={"query_text": query},
                timeout=(10, 180),
            )
            resp.raise_for_status()
            raw_answer = resp.json().get("answer", "")

            answer, citations = parse_answer(raw_answer)

            # ── Display Answer ───────────────────────────────
            st.markdown("---")
            st.subheader("Answer")

            is_fallback = "I don't know based on the provided context" in answer
            if is_fallback:
                st.warning(answer)
            else:
                st.write(answer)

            # ── Display Citations ────────────────────────────
            if citations:
                st.subheader("Sources")
                for i, c in enumerate(citations, 1):
                    source = escape(str(c.get("source", "unknown")))
                    page = escape(str(c.get("page", "N/A")))
                    page_text = f"  ·  page {page}" if page not in ("N/A", -1, "-1") else ""
                    st.markdown(
                        f'<span class="citation-badge">📄 [{i}] {source}{page_text}</span>',
                        unsafe_allow_html=True,
                    )

            # ── Raw JSON (collapsed) ─────────────────────────
            with st.expander("Show raw response"):
                st.code(raw_answer, language="json")

        except requests.Timeout:
            st.warning("The answer took too long. Please wait before trying again.")
        except requests.ConnectionError:
            st.error(
                "**Cannot reach backend.** "
                "Make sure `python server.py` is running on port 8000."
            )
        except requests.HTTPError:
            try:
                detail = resp.json().get("detail", "Your question could not be answered.")
            except ValueError:
                detail = "Your question could not be answered."
            st.error(detail)
        except Exception as e:
            st.error(f"Unexpected error: {e}")

elif ask:
    st.warning("Please enter a question first.")
