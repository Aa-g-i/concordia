"""Unit tests for SchedulingEnvironment."""

import unittest
import tempfile
import os
import time
from concordia.environment.scheduling_environment import SchedulingEnvironment, FixedStepClock, EventDrivenClock
from concordia.typing import entity as entity_lib

class MockAgent:
  def __init__(self, name, preset_responses):
    self.name = name
    self.preset_responses = preset_responses
    self.call_count = 0
    self.next_wakeup_time = 0.0
    self.observations = []
    
  def act(self, call_to_action: str):
    if self.call_count < len(self.preset_responses):
      resp = self.preset_responses[self.call_count]
      self.call_count += 1
      return resp
    return "wait: 15"
    
  def observe(self, event: str):
    self.observations.append(event)

class SchedulingEnvironmentTest(unittest.TestCase):

  def test_basic_progression(self):
    strategy = FixedStepClock(time_step=5.0)
    alice = MockAgent("Alice", ["wait: 10", "speak: hello"])
    
    env = SchedulingEnvironment(timing_strategy=strategy, start_time=0.0, end_time=20.0)
    
    # Run loop
    env.run_loop(game_masters=[], entities=[alice], premise="", max_steps=10, verbose=False, log=[])
    
    # Verify clock advanced
    self.assertGreaterEqual(env._clock_time, 20.0)
    # Verify Alice spoke
    self.assertIn("[15.0] Alice: hello", env._chat_history)

  def test_heartbeat_file(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      alice = MockAgent("Alice", ["wait: 10"])
      strategy = FixedStepClock(time_step=5.0)
      env = SchedulingEnvironment(timing_strategy=strategy, start_time=0.0, end_time=10.0, out_dir=tmpdir)
      
      env.run_loop(game_masters=[], entities=[alice], premise="", max_steps=10, verbose=False, log=[])
      
      heartbeat_path = os.path.join(tmpdir, "heartbeat.txt")
      self.assertTrue(os.path.exists(heartbeat_path))

  def test_timeout_behavior(self):
    alice = MockAgent("Alice", ["wait: 10"])
    strategy = FixedStepClock(time_step=1.0)
    env = SchedulingEnvironment(timing_strategy=strategy, start_time=0.0, end_time=100.0, max_real_time_seconds=0.001)
    
    def slow_act(prompt):
      time.sleep(0.01)
      return "wait: 15"
    alice.act = slow_act
    
    env.run_loop(game_masters=[], entities=[alice], premise="", max_steps=10, verbose=False, log=[])
    self.assertLess(env._clock_time, 100.0)

  def test_reactive_turns_cap(self):
    alice = MockAgent("Alice", ["speak: hello"] * 10)
    bob = MockAgent("Bob", ["speak: hi"] * 10)
    strategy = FixedStepClock(time_step=10.0)
    
    env = SchedulingEnvironment(timing_strategy=strategy, start_time=0.0, end_time=20.0)
    env.run_loop(game_masters=[], entities=[alice, bob], premise="", max_steps=10, verbose=False, log=[])
    
    self.assertLessEqual(len(env._chat_history), 16)

  def test_wake_up_on_speech(self):
    alice = MockAgent("Alice", ["speak: i am awake"])
    alice.next_wakeup_time = 100.0
    bob = MockAgent("Bob", ["speak: wake up"])
    strategy = FixedStepClock(time_step=5.0)
    
    env = SchedulingEnvironment(timing_strategy=strategy, start_time=0.0, end_time=15.0)
    env.run_loop(game_masters=[], entities=[bob, alice], premise="", max_steps=10, verbose=False, log=[])
    
    # Bob speaks at 5.0, wakes Alice up, Alice speaks at 10.0 (due to latency pushing her to next tick).
    self.assertIn("[05.0] Bob: wake up", env._chat_history)
    self.assertIn("[10.0] Alice: i am awake", env._chat_history)

  def test_valid_json_action(self):
    strategy = FixedStepClock(time_step=5.0)
    alice = MockAgent("Alice", ['{"action": "speak", "speech": "hello JSON"}'])
    env = SchedulingEnvironment(timing_strategy=strategy, start_time=0.0, end_time=20.0)
    
    env.run_loop(game_masters=[], entities=[alice], premise="", max_steps=10, verbose=False, log=[])
    self.assertIn("[05.0] Alice: hello JSON", env._chat_history)

  def test_invalid_json_fallback(self):
    strategy = FixedStepClock(time_step=5.0)
    alice = MockAgent("Alice", [])
    env = SchedulingEnvironment(timing_strategy=strategy, start_time=0.0, end_time=20.0)
    env._entities = [alice]
    
    # Test invalid JSON
    env.resolve(alice, '{"action": "speak", "speech": "hello"')
    # Should hit unknown action and set default sleep +15
    self.assertEqual(alice.next_wakeup_time, 15.0)
    
    # Test valid JSON lacking 'action'
    env.resolve(alice, '{"not_action": "wait"}')
    self.assertEqual(alice.next_wakeup_time, 15.0)

  def test_latency_bounds(self):
    strategy = FixedStepClock(time_step=5.0)
    alice = MockAgent("Alice", [])
    bob = MockAgent("Bob", [])
    bob.next_wakeup_time = 100.0
    
    env = SchedulingEnvironment(timing_strategy=strategy, start_time=5.0, end_time=20.0)
    env._entities = [alice, bob] # Set entities manually for resolve
    
    # Alice speaks at t=5.0
    env.resolve(alice, "speak: hello")
    
    # Bob should be woken up with latency
    latency = bob.next_wakeup_time - 5.0
    self.assertGreaterEqual(latency, 0.5)
    self.assertLessEqual(latency, 1.5)

  def test_concurrent_wakeups(self):
    strategy = FixedStepClock(time_step=5.0)
    alice = MockAgent("Alice", ["speak: i am alice"])
    bob = MockAgent("Bob", ["speak: i am bob"])
    
    alice.next_wakeup_time = 5.0
    bob.next_wakeup_time = 5.0
    
    env = SchedulingEnvironment(timing_strategy=strategy, start_time=0.0, end_time=20.0)
    env.run_loop(game_masters=[], entities=[alice, bob], premise="", max_steps=10, verbose=False, log=[])
    
    # One will be picked first and push the other to next tick due to latency
    # So we expect one at 05.0 and the other at 10.0!
    # We don't know who goes first due to shuffle, so we check for presence of both at any allowed time
    history_str = "".join(env._chat_history)
    self.assertTrue("Alice: i am alice" in history_str)
    self.assertTrue("Bob: i am bob" in history_str)
    self.assertTrue("[05.0]" in history_str)
    self.assertTrue("[10.0]" in history_str)

  def test_event_driven_clock_event_lockon(self):
    strategy = EventDrivenClock(min_step=10.0, max_jump=60.0)
    alice = MockAgent("Alice", [])
    alice.next_wakeup_time = 100.0
    
    class MockEvent:
      def __init__(self, timestamp):
        self.timestamp = timestamp
    
    events = [MockEvent(5.0)]
    next_time = strategy.get_next_tick(current_time=0.0, end_time=20.0, agents=[alice], historical_events=events)
    self.assertEqual(next_time, 5.0)

  def test_event_driven_clock_max_jump(self):
    strategy = EventDrivenClock(min_step=1.0, max_jump=3.0)
    alice = MockAgent("Alice", [])
    alice.next_wakeup_time = 100.0
    
    next_time = strategy.get_next_tick(current_time=0.0, end_time=20.0, agents=[alice], historical_events=[])
    self.assertEqual(next_time, 3.0)

if __name__ == '__main__':
  unittest.main()


