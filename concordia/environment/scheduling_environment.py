"""Alternative to discrete timestep engine which lets actors broadcast the next time they're speak to other participants"""

import abc
from collections.abc import Callable, Mapping, Sequence
import json
import os
import random
import time
from typing import Any

from absl import logging
from concordia.typing import entity as entity_lib
from concordia.environment import engine as engine_lib

class TimingStrategy(abc.ABC):
  @abc.abstractmethod
  def get_next_tick(self, current_time: float, end_time: float, agents: Sequence[entity_lib.Entity], historical_events: list) -> float:
    pass

class FixedStepClock(TimingStrategy):
  def __init__(self, time_step: float = 1.0):
    self.time_step = time_step

  def get_next_tick(self, current_time: float, end_time: float, agents: Sequence[entity_lib.Entity], historical_events: list) -> float:
    return min(end_time, current_time + self.time_step)

class EventDrivenClock(TimingStrategy):
  def __init__(self, min_step: float = 1.0, max_jump: float = 3.0):
    self.min_step = min_step
    self.max_jump = max_jump

  def get_next_tick(self, current_time: float, end_time: float, agents: Sequence[entity_lib.Entity], historical_events: list) -> float:
    next_time = end_time
    found_agent = False
    for agent in agents:
      if hasattr(agent, 'next_wakeup_time'):
        if agent.next_wakeup_time > current_time:
          next_time = min(next_time, agent.next_wakeup_time)
          found_agent = True
          
    # Target historical events
    next_event_times = [e.timestamp for e in historical_events if hasattr(e, 'timestamp') and e.timestamp > current_time]
    if next_event_times:
      next_time = min(next_time, min(next_event_times))
      
    if not found_agent and not next_event_times:
      next_time = min(end_time, current_time + self.min_step)
      
    # Enforce max jump to prevent long silences
    next_time = min(next_time, current_time + self.max_jump)
      
    return next_time

class SchedulingEnvironment(engine_lib.Engine):
  """A scheduling environment that runs based on a virtual clock and event timings, allowing immediate reactions within the same tick."""

  def __init__(
      self,
      timing_strategy: TimingStrategy,
      start_time: float = 0.0,
      end_time: float = 100.0,
      max_real_time_seconds: float = 900.0,
      out_dir: str | None = None,
      stage_id: str | None = None,
  ):
    self._timing_strategy = timing_strategy
    self._clock_time = start_time
    self._end_time = end_time
    self._max_real_time_seconds = max_real_time_seconds
    self._out_dir = out_dir
    self._stage_id = stage_id
    self._chat_history = []
    self._last_broadcast_time = start_time - 0.001
    self._events = []
    
    # State for current tick
    self._current_tick_agents = []
    self._acted_in_current_tick = set()
    self._spoke_this_tick = False
    self._reactive_turns = 0
    self._start_real_time = None

  def set_events(self, events):
    self._events = sorted(events, key=lambda e: e.timestamp)

  def make_observation(
      self,
      game_master: entity_lib.Entity,
      entity: entity_lib.Entity,
  ) -> str:
    # Broadcast events that occurred between last broadcast and current clock time
    observations = []
    for event in self._events:
      if self._last_broadcast_time < event.timestamp <= self._clock_time:
        if hasattr(event, 'content'):
          observations.append(str(event.content))
        else:
          observations.append(str(event))
          
    # Wake up sleeping agents on new events
    if observations and hasattr(entity, 'next_wakeup_time'):
      entity.next_wakeup_time = self._clock_time
          
    return "\n".join(observations)

  def next_acting(
      self,
      game_master: entity_lib.Entity,
      entities: Sequence[entity_lib.Entity],
  ) -> tuple[entity_lib.Entity, entity_lib.ActionSpec] | None:
    
    if not self._current_tick_agents:
      # Start of a new tick
      self._current_tick_agents = list(entities)
      random.shuffle(self._current_tick_agents)
      self._acted_in_current_tick.clear()
      
    for entity in self._current_tick_agents:
      if entity not in self._acted_in_current_tick:
        # Skip polling if the entity is asleep
        if hasattr(entity, 'next_wakeup_time') and entity.next_wakeup_time > self._clock_time:
          self._acted_in_current_tick.add(entity)
          continue
          
        self._acted_in_current_tick.add(entity)
        
        # Generalize to ActionSpec as requested
        call_to_action = (
            f"[Time: {self._clock_time}]. State your next intention. "
            "Respond with 'wait' or 'speak'. If you wait, specify delay like 'wait: 15'. "
            "If you speak, specify message like 'speak: hello'."
        )
        return entity, entity_lib.ActionSpec(
            call_to_action=call_to_action,
            output_type=entity_lib.OutputType.FREE,
        )
        
    # Everyone has acted in this tick
    return None

  def resolve(
      self,
      game_master: entity_lib.Entity,
      event: str,
  ) -> None:
    # Event is the raw response from the agent. We need to parse it.
    self._acted_in_current_tick.add(game_master)
    
    parsed_action = None
    parsed_speech = None
    parsed_delay = None
    
    # Try JSON parsing first
    try:
      start_idx = event.find('{')
      end_idx = event.rfind('}')
      if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        json_str = event[start_idx:end_idx+1]
        data = json.loads(json_str)
        parsed_action = data.get('action')
        parsed_speech = data.get('speech')
        parsed_delay = data.get('delay')
    except Exception as e:
      logging.debug(f"Failed to parse JSON block from event: {e}")
      
    # Fallback to string parsing if JSON was not present or failed to yield an action
    if not parsed_action:
      if event.startswith("speak:"):
        parsed_action = "speak"
        parsed_speech = event[len("speak:"):].strip()
      elif event.startswith("wait:"):
        parsed_action = "wait"
        delay_str = event[len("wait:"):].strip()
        try:
          parsed_delay = float(delay_str)
        except ValueError:
          parsed_delay = 15.0 # Fallback
          
    # Execute the parsed action
    try:
      if parsed_action == "speak":
        message = parsed_speech or ""
        log_entry = f"[{self._clock_time:04.1f}] {game_master.name}: {message}"
        self._chat_history.append(log_entry)
        print(log_entry)
        self._spoke_this_tick = True
        
        # Broadcast this speech to other entities immediately and wake them up
        for other_entity in self._entities:
          if other_entity != game_master and hasattr(other_entity, 'observe'):
            other_entity.observe(log_entry)
            if hasattr(other_entity, 'next_wakeup_time'):
              # Add a small randomized latency (0.5-1.5s) to simulate human reaction time
              other_entity.next_wakeup_time = self._clock_time + random.uniform(0.5, 1.5)
              
      elif parsed_action == "wait":
        delay = parsed_delay if parsed_delay is not None else 15.0
        game_master.next_wakeup_time = self._clock_time + delay
      else:
        logging.warning(f"Unknown action parsed from event: {event}")
        # Default sleep
        game_master.next_wakeup_time = self._clock_time + 15.0
        
    except Exception as e:
      logging.warning(f"Error executing action: {e}")
      game_master.next_wakeup_time = self._clock_time + 15.0

  def terminate(
      self,
      game_master: entity_lib.Entity,
  ) -> bool:
    if self._clock_time >= self._end_time:
      return True
    if self._start_real_time and (time.time() - self._start_real_time > self._max_real_time_seconds):
      logging.warning(f"Timeout reached ({self._max_real_time_seconds}s)")
      return True
    return False

  def next_game_master(
      self,
      game_master: entity_lib.Entity,
      game_masters: Sequence[entity_lib.Entity],
  ) -> entity_lib.Entity:
    return game_master

  def run_loop(
      self,
      game_masters: Sequence[entity_lib.Entity],
      entities: Sequence[entity_lib.Entity],
      premise: str,
      max_steps: int,
      verbose: bool,
      log: list[Mapping[str, Any]] | None,
      checkpoint_callback: Callable[[int], None] | None = None,
      step_controller: Any = None,
      step_callback: Callable[[Any], None] | None = None,
  ):
    self._entities = entities
    print(f"--- Starting chronological simulation loop from t={self._clock_time} to t={self._end_time} ---")
    self._start_real_time = time.time()
    
    while not self.terminate(None):
      # Advance clock via strategy
      next_time = self._timing_strategy.get_next_tick(self._clock_time, self._end_time, entities, self._events)
      
      if next_time <= self._clock_time:
        next_time = self._clock_time + 1.0
        
      self._clock_time = next_time
      
      # Update virtual clocks for deterministic models if any
      for entity in entities:
        if hasattr(entity, 'set_clock'):
          entity.set_clock(self._clock_time)
          
      # Make observations (push events)
      for entity in entities:
        obs = self.make_observation(None, entity)
        if obs and hasattr(entity, 'observe'):
          entity.observe(obs)
          
      self._last_broadcast_time = self._clock_time
      
      self._spoke_this_tick = True
      self._reactive_turns = 0
      
      while self._spoke_this_tick and self._reactive_turns < 7:
        self._spoke_this_tick = False
        self._current_tick_agents = [] # Reset candidate list to force shuffle and re-poll
        
        while True:
          acting = self.next_acting(None, entities)
          if acting is None:
            break
            
          entity, action_spec = acting
          
          # Poll entity
          if hasattr(entity, 'act'):
            response = entity.act(action_spec.call_to_action)
            self.resolve(entity, response)
            
        if self._spoke_this_tick:
          self._reactive_turns += 1
          
      print(f"[{self._stage_id or ''} t={self._clock_time}] Tick completed.")
      
      # Heartbeat file
      if self._out_dir:
        heartbeat_file = os.path.join(self._out_dir, "heartbeat.txt")
        try:
          with open(heartbeat_file, "w") as f:
            f.write(f"t={self._clock_time}\n")
        except Exception:
          pass
          
    print("--- Simulation loop finished ---")
    return self._chat_history
