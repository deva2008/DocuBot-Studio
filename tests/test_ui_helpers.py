import os
import unittest
import streamlit as st

from app.utils.ui_helpers import get_openai_api_key


class TestUIHelpers(unittest.TestCase):
    def setUp(self):
        # backup env and session
        self._orig_env = os.environ.get("OPENAI_API_KEY")
        self._had_session_key = "OPENAI_API_KEY" in st.session_state
        self._orig_session = st.session_state.get("OPENAI_API_KEY")
        # ensure clean
        if self._had_session_key:
            st.session_state.pop("OPENAI_API_KEY", None)
        os.environ.pop("OPENAI_API_KEY", None)

    def tearDown(self):
        # restore
        if self._had_session_key and self._orig_session is not None:
            st.session_state["OPENAI_API_KEY"] = self._orig_session
        else:
            st.session_state.pop("OPENAI_API_KEY", None)
        if self._orig_env is not None:
            os.environ["OPENAI_API_KEY"] = self._orig_env
        else:
            os.environ.pop("OPENAI_API_KEY", None)

    def test_returns_none_when_no_key(self):
        st.session_state.pop("OPENAI_API_KEY", None)
        os.environ.pop("OPENAI_API_KEY", None)
        self.assertIsNone(get_openai_api_key())

    def test_prefers_session_over_env(self):
        os.environ["OPENAI_API_KEY"] = "env-key-1234567890"
        st.session_state["OPENAI_API_KEY"] = "session-key-abcdefg"
        self.assertEqual(get_openai_api_key(), "session-key-abcdefg")

    def test_uses_env_when_no_session(self):
        os.environ["OPENAI_API_KEY"] = "env-key-1234567890"
        st.session_state.pop("OPENAI_API_KEY", None)
        self.assertEqual(get_openai_api_key(), "env-key-1234567890")


if __name__ == "__main__":
    unittest.main()
