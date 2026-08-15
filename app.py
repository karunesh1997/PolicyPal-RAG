import shutil
from pathlib import Path

import streamlit as st

from config.config import (
    DOCUMENTS_DIR,
    TOP_K,
    GEMINI_MODEL
)

from src.pipeline import PolicyPipeline

from src.utils.helpers import (
    get_pdf_files,
    confidence_label
)


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="PolicyPal",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown(
    """
    <style>

    .main {
        background-color: #f8fafc;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
    }

    .hero {
        padding: 1.5rem;
        border-radius: 18px;
        background: linear-gradient(
            135deg,
            #111827,
            #1e3a8a
        );
        color: white;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        font-size: 2.2rem;
        margin-bottom: 0.2rem;
    }

    .hero p {
        color: #dbeafe;
        margin-bottom: 0;
    }

    .source-card {
        padding: 0.9rem;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        background: white;
        margin-bottom: 0.7rem;
    }

    .metric-card {
        padding: 1rem;
        border-radius: 14px;
        background: white;
        border: 1px solid #e5e7eb;
        text-align: center;
    }

    .small-text {
        color: #6b7280;
        font-size: 0.85rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []


if "pipeline" not in st.session_state:

    st.session_state.pipeline = None


# ---------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------

@st.cache_resource
def load_pipeline():

    return PolicyPipeline()


pipeline = load_pipeline()


# ---------------------------------------------------------
# HERO
# ---------------------------------------------------------

st.markdown(
    """
    <div class="hero">

        <h1>🏢 PolicyPal</h1>

        <p>
        Your AI-powered company policy assistant.
        Ask questions and get source-grounded answers
        from your organization's documents.
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.header("📚 Policy Library")

    uploaded_files = st.file_uploader(
        "Upload policy PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:

        for uploaded_file in uploaded_files:

            destination = (
                DOCUMENTS_DIR /
                uploaded_file.name
            )

            with open(
                destination,
                "wb"
            ) as file:

                file.write(
                    uploaded_file.getbuffer()
                )

        st.success(
            f"{len(uploaded_files)} document(s) uploaded."
        )

    st.divider()

    pdf_files = get_pdf_files(
        DOCUMENTS_DIR
    )

    st.subheader(
        f"Documents ({len(pdf_files)})"
    )

    if pdf_files:

        for pdf in pdf_files:

            st.markdown(
                f"📄 `{pdf.name}`"
            )

    else:

        st.info(
            "Upload PDF policies to get started."
        )

    st.divider()

    st.subheader("⚙️ Knowledge Base")

    if st.button(
        "🔄 Build / Rebuild Knowledge Base",
        use_container_width=True
    ):

        if not pdf_files:

            st.error(
                "Please upload at least one PDF."
            )

        else:

            with st.spinner(
                "Processing policies..."
            ):

                try:

                    chunks = pipeline.build_index()

                    st.success(
                        f"Knowledge base ready — "
                        f"{chunks} chunks indexed."
                    )

                except Exception as exc:

                    st.error(
                        f"Indexing failed: {exc}"
                    )

    if pipeline.is_ready():

        st.success(
            "🟢 Knowledge base ready"
        )

    else:

        st.warning(
            "🟡 Knowledge base not built"
        )

    st.divider()

    st.caption(
        f"AI Model: {GEMINI_MODEL}"
    )

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ---------------------------------------------------------
# MAIN CONTENT
# ---------------------------------------------------------

if not pipeline.is_ready():

    st.info(
        "👈 Upload your policy documents and "
        "build the knowledge base to begin."
    )

    st.subheader(
        "Try asking questions like:"
    )

    example_columns = st.columns(2)

    examples = [
        "How many casual leaves can employees take?",
        "What is the work-from-home policy?",
        "What is the travel reimbursement limit?",
        "What is the notice period?",
    ]

    for column, question in zip(
        example_columns,
        examples
    ):

        with column:

            st.markdown(
                f"**→ {question}**"
            )

    st.stop()


# ---------------------------------------------------------
# DASHBOARD METRICS
# ---------------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown(
        f"""
        <div class="metric-card">

        <h3>{len(pdf_files)}</h3>

        <div class="small-text">
        Policy Documents
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

with col2:

    st.markdown(
        """
        <div class="metric-card">

        <h3>RAG</h3>

        <div class="small-text">
        Retrieval Engine
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

with col3:

    st.markdown(
        """
        <div class="metric-card">

        <h3>Local</h3>

        <div class="small-text">
        AI Processing
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


st.divider()


# ---------------------------------------------------------
# CHAT HISTORY
# ---------------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            st.markdown(
                "#### 📚 Sources"
            )

            for source in message["sources"]:

                score = source["score"]

                st.markdown(
                    f"""
                    <div class="source-card">

                    <strong>
                    📄 {source['source']}
                    </strong>

                    <br>

                    Page {source['page']}

                    <br>

                    <span class="small-text">
                    Retrieval score: {score:.2f}
                    </span>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


# ---------------------------------------------------------
# CHAT INPUT
# ---------------------------------------------------------

question = st.chat_input(
    "Ask a question about your company policies..."
)


if question:

    # User message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)


    # Assistant
    with st.chat_message("assistant"):

        with st.spinner(
            "Searching policies..."
        ):

            try:

                result = pipeline.ask(
                    question,
                    top_k=TOP_K
                )

                answer = result["answer"]

                sources = result["sources"]

                confidence = result[
                    "confidence"
                ]

                st.markdown(answer)

                # Confidence
                st.markdown(
                    f"""
                    **Retrieval confidence:** 
                    {confidence_label(confidence)}
                    ({confidence:.2f})
                    """
                )

                if sources:

                    st.markdown(
                        "#### 📚 Sources"
                    )

                    for source in sources:

                        st.markdown(
                            f"""
                            <div class="source-card">

                            <strong>
                            📄 {source['source']}
                            </strong>

                            <br>

                            Page {source['page']}

                            <br>

                            <span class="small-text">
                            Retrieval score:
                            {source['score']:.2f}
                            </span>

                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "confidence": confidence
                    }
                )

            except Exception as exc:

                error_message = (
                    f"Something went wrong: {exc}"
                )

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message
                    }
                )