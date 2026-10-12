"""Streamlit page smoke tests; no provider calls or participant data."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest

class AppTests(unittest.TestCase):
    def test_roles_pages_and_test_lock(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'APP_PASSWORD':'synthetic-analyst-pass','ADJUDICATOR_PASSWORD':'synthetic-adjudicator-pass','NAJAH_DB_PATH':str(Path(tmp)/'test.sqlite')}):
            at=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run(timeout=20)
            self.assertEqual(len(at.exception),0)
            at.text_input[0].set_value('synthetic-analyst-pass')
            at.button[0].click().run(timeout=20)
            self.assertEqual(len(at.exception),0)
            for page in ['Data and splits','Reference quality','Judge development','Freeze protocol','Locked test','Exports']:
                at.sidebar.radio[0].set_value(page).run(timeout=20)
                self.assertEqual(len(at.exception),0,page)
            at.sidebar.button[0].click().run(timeout=20)
            at.selectbox[0].set_value('Human adjudicator')
            at.text_input[0].set_value('synthetic-adjudicator-pass')
            at.button[0].click().run(timeout=20)
            self.assertEqual(list(at.sidebar.radio[0].options),['Human adjudication'])
            self.assertEqual(len(at.exception),0)

    def test_missing_passwords_fail_closed(self):
        with patch.dict(os.environ,{'APP_PASSWORD':'','ADJUDICATOR_PASSWORD':''}):
            at=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run(timeout=20)
            self.assertEqual(len(at.exception),0)
            self.assertEqual(len(at.error),1)

if __name__=='__main__':unittest.main()
