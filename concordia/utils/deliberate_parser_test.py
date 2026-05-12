"""Unit tests for deliberate_parser."""

import unittest
import tempfile
import os
from concordia.utils.deliberate_parser import load_chat_logs, ChatEvent

class DeliberateParserTest(unittest.TestCase):

    def test_load_chat_logs_standard(self):
        csv_content = """timestamp,participant_id,message
2025-11-18 00:05:19.000,Alice,Hello Bob!
2025-11-18 00:05:20.000,Bob,Hi Alice.
"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as temp:
            temp.write(csv_content)
            temp_path = temp.name
            
        try:
            events = load_chat_logs(temp_path)
            self.assertEqual(len(events), 2)
            self.assertEqual(events[0].participant_id, "Alice")
            self.assertEqual(events[0].message, "Hello Bob!")
            self.assertEqual(events[1].participant_id, "Bob")
            self.assertEqual(events[1].message, "Hi Alice.")
        finally:
            os.remove(temp_path)

    def test_load_chat_logs_different_columns(self):
        csv_content = """timestamp,sender id,text
2025-11-18 00:05:19.000,Alice,Hello Bob!
"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as temp:
            temp.write(csv_content)
            temp_path = temp.name
            
        try:
            events = load_chat_logs(temp_path)
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].participant_id, "Alice")
            self.assertEqual(events[0].message, "Hello Bob!")
        finally:
            os.remove(temp_path)

    def test_load_chat_logs_sorting(self):
        csv_content = """timestamp,participant_id,message
2025-11-18 00:05:20.000,Bob,Hi Alice.
2025-11-18 00:05:19.000,Alice,Hello Bob!
"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as temp:
            temp.write(csv_content)
            temp_path = temp.name
            
        try:
            events = load_chat_logs(temp_path)
            self.assertEqual(len(events), 2)
            # Should be sorted by timestamp
            self.assertEqual(events[0].participant_id, "Alice")
            self.assertEqual(events[1].participant_id, "Bob")
        finally:
            os.remove(temp_path)

if __name__ == '__main__':
    unittest.main()
