"""Tests the DeterministicParticipantModel with SchedulingEnvironment."""

import unittest
import json
import random
from unittest.mock import patch
from concordia.utils.deliberate_parser import ChatEvent, SurveyEvent
from concordia.deliberate_integration.deterministic_models import DeterministicParticipantModel
from concordia.environment.scheduling_environment import SchedulingEnvironment, FixedStepClock

class MockAgent:
    def __init__(self, name, events):
        self.name = name
        self.model = DeterministicParticipantModel(name, events)

    def act(self, prompt: str):
        return self.model.sample_text(prompt)

    def set_clock(self, time: float):
        self.model.set_clock(time)
        
    def observe(self, event: str):
        pass

class PlaybackTest(unittest.TestCase):

    @patch('random.uniform', return_value=0.0)
    def test_chronological_playback(self, mock_uniform):
        """Tests that agents correctly wait and speak at specific timestamped intervals."""
        events = [
            ChatEvent(timestamp=10.0, participant_id="Alice", message="Hello Bob!"),
            ChatEvent(timestamp=15.0, participant_id="Bob", message="Hi Alice. How are you?"),
            ChatEvent(timestamp=16.0, participant_id="Alice", message="I'm good."),
        ]

        alice = MockAgent("Alice", events)
        bob = MockAgent("Bob", events)
        entities = [alice, bob]

        strategy = FixedStepClock(time_step=1.0)
        env = SchedulingEnvironment(timing_strategy=strategy, start_time=0.0, end_time=20.0)

        history = env.run_loop(game_masters=[], entities=entities, premise="", max_steps=100, verbose=False, log=[])

        # Verify history
        history_str = "".join(history)
        self.assertIn("[10.0] Alice: Hello Bob!", history_str)
        self.assertIn("[15.0] Bob: Hi Alice. How are you?", history_str)
        self.assertIn("[16.0] Alice: I'm good.", history_str)

    def test_survey_interception(self):
        """Tests that a SurveyEvent preempts the chronological chat logic if prompted."""
        events = [
            SurveyEvent(timestamp=5.0, participant_id="Charlie", survey_prompt="rank the issues", survey_response='{"rank": ["A"]}'),
            ChatEvent(timestamp=100.0, participant_id="Charlie", message="Chatting late.")
        ]

        agent = MockAgent("Charlie", events)

        # Standard chat prompt should return wait since we are at t=0
        agent.model.set_clock(0.0)
        standard_prompt = "What is your intention? Output JSON with 'intention' (wait|speak) and 'message'"
        resp = agent.act(standard_prompt)
        self.assertIn("wait", json.loads(resp).get("action"))

        # Survey prompt should intercept and return the survey data immediately, ignoring clock
        survey_prompt = "Private survey: please rank the issues you care about from highest to lowest."
        survey_resp = agent.act(survey_prompt)
        self.assertEqual(survey_resp, '{"rank": ["A"]}')

    def test_partial_agent_adapter_switch(self):
        """Tests that PartialAgentAdapter switches from playback to generative action at the exact timestamp."""
        from concordia.deliberate_integration.run_dl_experiment import PartialAgentAdapter
        
        events = [
            ChatEvent(timestamp=10.0, participant_id="Alice", message="Hello world!"),
            ChatEvent(timestamp=20.0, participant_id="Alice", message="自由讨论"),
        ]
        
        # We specify a partial_ratio of 0.5. Since we have 2 events, the global_switch_index is 1, and switch_time is 20.0.
        # Below switch_time (t <= 20.0), it should return deterministic playback.
        # Above switch_time (t > 20.0), it should call the generative inner model.
        
        class StubGenerativeAgent:
            def __init__(self, name, **kwargs):
                self.name = name
            def act(self, prompt):
                return json.dumps({"action": "speak", "speech": "I am an AI!"})
                
        adapter = PartialAgentAdapter(
            name="Alice",
            events=events,
            partial_ratio=0.5,
            inner_agent_class=StubGenerativeAgent,
            mode="simulacra_partial"
        )
        
        # 1. At t=10.0 (<= switch_time 20.0), should return the first event
        adapter.set_clock(10.0)
        resp1 = json.loads(adapter.act("What is your action?"))
        self.assertEqual(resp1.get("action"), "speak")
        self.assertEqual(resp1.get("speech"), "Hello world!")
        
        # 2. At t=30.0 (> switch_time 20.0), should switch to StubGenerativeAgent and return AI speech
        adapter.set_clock(30.0)
        resp2 = json.loads(adapter.act("What is your action?"))
        self.assertEqual(resp2.get("action"), "speak")
        self.assertEqual(resp2.get("speech"), "I am an AI!")

    def test_mask_or_mirror_paper_agent_deterministic(self):
        """Tests that MaskOrMirrorPaperAgent correctly initializes and acts in deterministic mode."""
        from concordia.deliberate_integration.run_dl_experiment import MaskOrMirrorPaperAgent
        
        events = [
            ChatEvent(timestamp=5.0, participant_id="Alice", message="Paper agent speaks.")
        ]
        agent = MaskOrMirrorPaperAgent(name="Alice", events=events, mode="deterministic")
        agent.model.set_clock(5.0)
        
        resp = json.loads(agent.act("Act"))
        self.assertEqual(resp.get("action"), "speak")
        self.assertEqual(resp.get("speech"), "Paper agent speaks.")

    def test_no_dialogue_agent_adapter_behavior(self):
        """Tests NoDialogueAgentAdapter routing during chat stages (playback) and non-chat stages (generative)."""
        from concordia.deliberate_integration.run_dl_experiment import NoDialogueAgentAdapter
        
        events = [
            ChatEvent(timestamp=5.0, participant_id="Alice", message="Historical chat.")
        ]
        
        class StubGenerativeAgent:
            def __init__(self, name, **kwargs):
                self.name = name
            def act(self, prompt):
                return json.dumps({"action": "speak", "speech": prompt})
                
        # 1. Test in 'mirror' mode
        adapter = NoDialogueAgentAdapter(
            name="Alice",
            events=events,
            mask_or_mirror="mirror",
            inner_agent_class=StubGenerativeAgent,
            run_mode="nodialogue_mirror"
        )
        
        # Discussion stage configuration
        adapter.set_stage("stage_1", {"kind": "chat", "name": "General discussion"})
        adapter.set_clock(5.0)
        
        # During chat stages, it should return deterministic playback
        resp_chat = json.loads(adapter.act("Act"))
        self.assertEqual(resp_chat.get("action"), "speak")
        self.assertEqual(resp_chat.get("speech"), "Historical chat.")
        
        # Non-discussion stage configuration
        adapter.set_stage("stage_2", {"kind": "survey", "name": "Demographic survey"})
        
        # During non-chat stages, it should route to generative model and inject optimal system instruction
        resp_gen = json.loads(adapter.act("Answer Survey"))
        self.assertEqual(resp_gen.get("action"), "speak")
        self.assertIn("SYSTEM INSTRUCTION: You should behave optimally in spite of human bias.", resp_gen.get("speech"))

if __name__ == '__main__':
    unittest.main()


