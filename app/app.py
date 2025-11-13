import streamlit as st
from dotenv import load_dotenv
from utils.logger import get_logger
from utils.pdf_utils import load_pdfs
from utils.retriever_utils import build_retriever, retrieve_chunks
from utils.generator_utils import generate_answer
 
load_dotenv()
st.set_page_config(page_title="DocuBot Studio", page_icon="✈️", layout="wide")
logger = get_logger(__name__)


def main():
    st.title("DocuBot Studio")
    st.caption("Transform documentation into AI assistants.")

    uploaded_files = st.file_uploader(
        "Upload HR policy PDFs", type=["pdf"], accept_multiple_files=True
    )
    if "retriever" not in st.session_state:
        st.session_state["retriever"] = None

    col1, col2 = st.columns(2)
    with col1:
        build_clicked = st.button("Build index")
    with col2:
        query = st.text_input("Ask a question about HR policies")

    if build_clicked and uploaded_files:
        docs = load_pdfs(uploaded_files)
        retriever = build_retriever(docs)
        st.session_state["retriever"] = retriever
        st.success("Index built from uploaded PDFs.")

    if st.button("Ask") and query:
        retriever = st.session_state.get("retriever")
        if retriever is None:
            st.warning("Please build the index first using uploaded PDFs.")
        else:
            contexts = retrieve_chunks(retriever, query)
            answer, sources = generate_answer(query, contexts)
            st.subheader("Answer")
            st.write(answer)
            with st.expander("Sources / Context"):
                for s in sources:
                    st.write(s[:500] + ("..." if len(s) > 500 else ""))


if __name__ == "__main__":
    main()
