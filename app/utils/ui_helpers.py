# ui_helpers.py
# Reusable UI helpers for Airlines_QA_Bot/app
# - OpenAI API key widget (session + optional .env persistence)
# - get_openai_api_key() helper for embedding/LLM functions

import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv, set_key

_ENV_PATH = Path(".env")


def _load_env_if_present():
    if _ENV_PATH.exists():
        load_dotenv(dotenv_path=_ENV_PATH)


def _get_key_from_session_or_env():
    # prefer session-state (entered) then process env
    key = st.session_state.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        _load_env_if_present()
        key = os.getenv("OPENAI_API_KEY")
    return key


def _set_key_for_session(key: str):
    st.session_state["OPENAI_API_KEY"] = key
    os.environ["OPENAI_API_KEY"] = key


def _persist_key_to_dotenv(key: str):
    # explicit persistence (insecure plaintext) - only do with user consent
    if not _ENV_PATH.exists():
        _ENV_PATH.write_text("")
    set_key(str(_ENV_PATH), "OPENAI_API_KEY", key)
    load_dotenv(dotenv_path=_ENV_PATH)
    os.environ["OPENAI_API_KEY"] = key


def _clear_session_key():
    st.session_state.pop("OPENAI_API_KEY", None)
    os.environ.pop("OPENAI_API_KEY", None)


def openai_api_key_widget(show_demo_toggle: bool = True):
    """
    Render the OpenAI API key UI. Call this in Step 3.
    Stores key in st.session_state["OPENAI_API_KEY"] or persists to local .env on demand.
    """
    st.markdown("### OpenAI API Key (optional)")
    with st.expander("Enter OpenAI API key (used for OpenAI embeddings & LLMs)"):
        col1, col2 = st.columns([4, 1])
        # default value: session or env masked
        default_val = st.session_state.get("OPENAI_API_KEY", "")
        key_input = col1.text_input(
            "Paste your OpenAI API key",
            value=default_val,
            placeholder="sk-...",
            type="password",
            key="ui_openai_key_input",
        )
        if col2.button("Save for session"):
            if not key_input:
                st.warning("Please paste a key before saving.")
            else:
                _set_key_for_session(key_input.strip())
                st.success("Key saved for this session (not persisted).")

        st.write("")  # small spacing
        persist = st.checkbox(
            "Persist key to local .env (insecure, plaintext)", value=False, key="persist_choice"
        )
        if persist:
            st.warning("Persisting stores the key on disk in plaintext. Only enable on a trusted machine.")
            if st.button("Persist key to .env"):
                if not key_input:
                    st.warning("No key to persist. Paste a key above.")
                else:
                    _persist_key_to_dotenv(key_input.strip())
                    st.success("Key written to .env and loaded into session.")

        if st.button("Clear session key"):
            _clear_session_key()
            st.info("Key cleared from session.")

        # optional demo toggle
        if show_demo_toggle:
            st.write("---")
            demo = st.checkbox(
                "Enable demo mode (use an ephemeral demo key) — DO NOT USE IN PRODUCTION",
                value=False,
                key="demo_choice",
            )
            if demo:
                # small harmless placeholder - replace with your own demo key mechanism if you have one
                demo_key = "sk-demo-please-replace"
                _set_key_for_session(demo_key)
                st.info("Demo mode enabled: demo key loaded into session (replace this with a secure demo token if desired).")

        # status / small format check
        loaded = _get_key_from_session_or_env()
        if loaded:
            looks_ok = loaded.startswith("sk-") and len(loaded) > 20
            st.write("Status:", "✅ Key loaded" if looks_ok else "⚠️ Key found (format looks odd)")
            masked = loaded[:4] + "..." + loaded[-4:]
            st.caption(f"Loaded key (masked): {masked}")
        else:
            st.info("No OpenAI key found. If you plan to use OpenAI embeddings/LLMs, enter your key above and 'Save for session'.")


def get_openai_api_key():
    """
    Use this function from embedding/LLM wrappers to obtain the user-supplied key.
    Returns the key string or None.
    """
    return _get_key_from_session_or_env()


# EOF
