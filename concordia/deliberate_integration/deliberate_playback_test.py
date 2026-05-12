"""Tests the Level 0 DeliberateLab -> Concordia playback rig."""

import unittest
import csv
import os
import tempfile
from collections.abc import Sequence

from concordia.agents import entity_agent
from concordia.components.agent import action_spec_ignored, memory, observation, plan
from concordia.associative_memory import basic_associative_memory

from concordia.utils.deliberate_parser import load_chat_logs, ChatEvent

from concordia.deliberate_integration.deterministic_models import DeterministicParticipantModel


class DeliberatePlaybackTest(unittest.TestCase):
    def setUp(self):
        # Create a mocked DeliberateLab dataset CSV for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.csv_path = os.path.join(self.temp_dir.name, "chat_logs.csv")
        with open(self.csv_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'participant_id', 'message'])
            writer.writerow(['1.0', 'P1', 'Hello everyone!'])
            writer.writerow(['2.0', 'P2', 'Hi P1. I think we should focus on the budget.'])
            writer.writerow(['3.0', 'P1', 'Agreed.'])

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_level_0_playback(self):
        # 1. Parse DL Mock Data Layer
        events = load_chat_logs(self.csv_path)
        self.assertEqual(len(events), 3)

        # 2. Build Deterministic Models (Level 0 Rig Pass-Through)
        model_p1 = DeterministicParticipantModel('P1', events)
        model_p2 = DeterministicParticipantModel('P2', events)
        import json
        
        # 3. Validation / Simulation Step Checking
        model_p1.set_clock(1.0)
        self.assertEqual(json.loads(model_p1.sample_text("Act"))["speech"], "Hello everyone!")
        
        model_p2.set_clock(2.0)
        self.assertEqual(json.loads(model_p2.sample_text("Act"))["speech"], "Hi P1. I think we should focus on the budget.")
        
        model_p1.set_clock(3.0)
        self.assertEqual(json.loads(model_p1.sample_text("Act"))["speech"], "Agreed.")
        
        # 4. Guarantee exact match trace termination (i.e., NO_OP fallback once CSV exhausts)
        self.assertEqual(json.loads(model_p2.sample_text("Act"))["action"], "wait")
        self.assertEqual(json.loads(model_p1.sample_text("Act"))["action"], "wait")

if __name__ == '__main__':
    unittest.main()
