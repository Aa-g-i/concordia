"""Dynamic wrapper for executing any DeliberateLab zip file chronologically into Concordia."""

import argparse
import tempfile
import zipfile
import os
import re
import traceback
import contextlib
import json

from concordia.deliberate_integration.stage_orchestrator import ExperimentOrchestrator
from concordia.deliberate_integration.deterministic_models import DeterministicParticipantModel
from concordia.environment.scheduling_environment import SchedulingEnvironment, EventDrivenClock, FixedStepClock
from concordia.deliberate_integration.generic_exporter import SimulationExporter
from concordia.contrib.language_models import language_model_setup
from concordia.associative_memory import basic_associative_memory
from concordia.typing.entity import ActionSpec, OutputType
import datetime
import numpy as np
import concurrent.futures
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception

def is_rate_limit(exception):
    err_str = str(exception).lower()
    err_type = type(exception).__name__.lower()
    return any(x in err_str or x in err_type for x in [
        "429", "quota", "rate", "resource", "exhausted",
        "502", "503", "504", "server error", "temporary",
        "resol", "connect", "timeout", "socket", "network"
    ])

class RetryingLanguageModelWrapper:
    def __init__(self, underlying_model):
        self._model = underlying_model
        # Use a massive worker pool so we can safely abandon hung network sockets without stalling the queue 
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=200)
        
    def __getattr__(self, name):
        return getattr(self._model, name)
        
    @retry(
        wait=wait_exponential(multiplier=2, min=5, max=120),
        stop=stop_after_attempt(15),
        retry=retry_if_exception(is_rate_limit)
    )
    def sample_text(self, *args, **kwargs) -> str:
        future = self._executor.submit(self._model.sample_text, *args, **kwargs)
        try:
            return future.result(timeout=120)
        except concurrent.futures.TimeoutError:
            raise TimeoutError("google-genai HTTP socket hung for 120s. Forcing tenacity retry.")

    @retry(
        wait=wait_exponential(multiplier=2, min=5, max=120),
        stop=stop_after_attempt(15),
        retry=retry_if_exception(is_rate_limit)
    )
    def sample_choice(self, *args, **kwargs):
        future = self._executor.submit(self._model.sample_choice, *args, **kwargs)
        try:
            return future.result(timeout=120)
        except concurrent.futures.TimeoutError:
            raise TimeoutError("google-genai HTTP socket hung for 120s. Forcing tenacity retry.")

class MaskOrMirrorPaperAgent:
    def __init__(self, name, events=None, mode="deterministic", api_key=None, demographic_context="", thinking=False, temperature=None, delay_seconds=0.0):
        self.name = name
        self.demographic_context = demographic_context
        self.context_history = []
        
        if mode == "deterministic":
            self.model = DeterministicParticipantModel(name, events or [])
        else:
            if not api_key:
                raise ValueError("An --api_key must be provided to run in Simulacra or Human LLM modes.")
            
            thinking_budget = 0 if not thinking else None
            
            self.model = RetryingLanguageModelWrapper(language_model_setup(
                api_type="gemini",
                model_name="models/gemini-3-flash-preview",
                api_key=api_key,
                disable_language_model=False,
                thinking_budget=thinking_budget,
                temperature=temperature,
                delay_seconds=delay_seconds
            ))
        
    def observe(self, event):
        if hasattr(event, 'content'):
            msg = str(event.content)
        elif hasattr(event, 'message'):
             msg = str(event.message)
        else:
            msg = str(event)
            
        # Filter out wait/speak intention JSON noise if broadcasted from GM
        if '{"intention":' in msg:
            return
            
        self.context_history.append(msg)
             
    def act(self, prompt: str):
        demo_prefix = f"YOUR PARTICIPANT PROFILE:\n{self.demographic_context}\n\n" if self.demographic_context else ""
        
        # Build discussion history similar to local simulacra layout
        discussion_content = "\n".join(self.context_history)
        discussion_history = f"--- Discussion History ---\n{discussion_content}\n--- End Discussion ---\n\n" if discussion_content else ""
        
        full_prompt = f"{demo_prefix}Background Context and Survey State for {self.name}:\n{discussion_history}Task:\n{prompt}"
        response = self.model.sample_text(full_prompt)
        
        # Robustly extract JSON object from potentially messy LLM outputs
        match = re.search(r'(\{[\s\S]*\})', response)
        if match:
            return match.group(1)
            
        return '{"intention": "wait", "delay_before_next_check_seconds": 15}'


class StagedAgentAdapter:
    def __init__(self, name, events=None, mode="deterministic", api_key=None, demographic_context="", thinking=False, temperature=None, delay_seconds=0.0):
        self.name = name
        self.demographic_context = demographic_context
        self.stage_history = {}
        self.current_stage = "Initialization"
        self.mode = mode
        
        if mode == "deterministic":
            self.model = DeterministicParticipantModel(name, events or [])
        else:
            if not api_key:
                raise ValueError("An --api_key must be provided to run in Simulacra or Human LLM modes.")
            
            thinking_budget = 0 if not thinking else None
            
            self.model = language_model_setup(
                api_type="gemini",
                model_name="models/gemini-3-flash-preview",
                api_key=api_key,
                disable_language_model=False,
                thinking_budget=thinking_budget,
                temperature=temperature,
                delay_seconds=delay_seconds
            )
            
    def set_stage(self, stage_id):
        self.current_stage = stage_id
        if self.current_stage not in self.stage_history:
            self.stage_history[self.current_stage] = []

    def observe(self, event):
        if hasattr(event, 'content'):
            msg = str(event.content)
        elif hasattr(event, 'message'):
             msg = str(event.message)
        else:
            msg = str(event)
            
        if '{"intention":' in msg:
            return
            
        if self.current_stage not in self.stage_history:
            self.stage_history[self.current_stage] = []
        self.stage_history[self.current_stage].append(msg)
             
    def act(self, prompt: str):
        demo_prefix = f"YOUR PARTICIPANT PROFILE:\n{self.demographic_context}\n\n" if self.demographic_context else ""
        
        context_blocks = []
        for stage, history in self.stage_history.items():
            if history:
                context_blocks.append(f"--- Stage: {stage} ---\n" + "\n".join(history))
        context_string = "\n\n".join(context_blocks)
        
        if self.mode == "human":
            instruction = "SYSTEM INSTRUCTION: You should strive to mirror (model the distribution of) the implicit human bias you'd expect for such a task."
        elif self.mode == "simulacra":
            instruction = "SYSTEM INSTRUCTION: You should strive to mask (compensate for) the implicit human bias you'd expect for such a task."
        else:
            instruction = ""
            
        full_prompt = f"{demo_prefix}{instruction}\n\nBackground Context and Event History for {self.name}:\n{context_string}\n\nTask:\n{prompt}"
        
        try:
            response = self.model.sample_text(full_prompt)
        except Exception as e:
            print(f"[{self.name}] LLM Context Overflow or API Failure in mode {self.mode}: {e}")
            return '{"intention": "wait", "delay_before_next_check_seconds": 15}'
        
        match = re.search(r'(\{[\s\S]*\})', response)
        if match:
            return match.group(1)
            
        return '{"intention": "wait", "delay_before_next_check_seconds": 15}'


class ConcordiaAgentAdapter:
    def __init__(self, name, events=None, mode="human", api_key=None, demographic_context="", thinking=False, temperature=None, delay_seconds=0.0):
        self.name = name
        
        if not api_key:
            raise ValueError("An --api_key must be provided to run Concordia LLM modes.")
            
        thinking_budget = 0 if not thinking else None
        
        model = RetryingLanguageModelWrapper(language_model_setup(
            api_type="gemini",
            model_name="models/gemini-3-flash-preview",
            api_key=api_key,
            disable_language_model=False,
            thinking_budget=thinking_budget,
            temperature=temperature,
            delay_seconds=delay_seconds
        ))
        
        setup_time = datetime.datetime.now()
        memories = basic_associative_memory.AssociativeMemoryBank(
            lambda x: np.zeros(384)
        )
        
        base_instruct = "**Important**: You are simulating the participant described in YOUR PARTICIPANT PROFILE. When making decisions, rely **only** on the perspective, knowledge, and lived experience of that persona. **Do not use outside knowledge, expert reasoning, or facts** beyond what the persona would plausibly know. Your reasoning should reflect the mindset and limitations of someone in this situation, not a general or expert perspective."
        
        if demographic_context:
            base_instruct = f"{base_instruct}\n\nYOUR PARTICIPANT PROFILE:\n{demographic_context}"

        if mode == "human":
            goal = f"{base_instruct}\n\n#### SIMULATION MODE: HUMAN\nMirror (model the distribution of) the implicit human bias you'd expect for such a task."
        elif mode == "simulacra":
            goal = f"{base_instruct}\n\n#### SIMULATION MODE: SIMULACRA\nMask (compensate for) the implicit human bias you'd expect for such a task."
        else:
            goal = base_instruct
            
        # Build a native Concordia Agent using standard components
        from concordia.agents import entity_agent_with_logging
        from concordia.components import agent as agent_components

        instructions = agent_components.instructions.Instructions(
            agent_name=name,
            pre_act_label='\nInstructions'
        )
        overarching_goal = agent_components.constant.Constant(
            state=goal,
            pre_act_label='\nGoal'
        )
        observation_to_memory = agent_components.observation.ObservationToMemory()
        observation = agent_components.observation.LastNObservations(
            history_length=50,
            pre_act_label='\nEvents so far (ordered from least recent to most recent)'
        )
        relevant_memories = agent_components.all_similar_memories.AllSimilarMemories(
            model=model,
            components=['LastNObservations'],
            num_memories_to_retrieve=100,
            pre_act_label='\nRecalled memories and observations'
        )
        self_perception = agent_components.question_of_recent_memories.SelfPerception(
            model=model,
            pre_act_label=f'\nQuestion: What kind of person is {name}?\nAnswer'
        )
        situation_perception = agent_components.question_of_recent_memories.SituationPerception(
            model=model,
            components=['AllSimilarMemories', 'SelfPerception'],
            pre_act_label='\nQuestion: What is the current situation?\nAnswer'
        )
        plan = agent_components.plan.Plan(
            model=model,
            components=['LastNObservations', 'AllSimilarMemories', 'SituationPerception'],
            goal_component_key='Goal',
            pre_act_label='\nPlan'
        )

        components_of_agent = {
            'Instructions': instructions,
            'Goal': overarching_goal,
            'Observation': observation_to_memory,
            'LastNObservations': observation,
            'AllSimilarMemories': relevant_memories,
            'Memory': agent_components.memory.AssociativeMemory(memory_bank=memories),
            'SelfPerception': self_perception,
            'SituationPerception': situation_perception,
            'Plan': plan
        }

        component_order = list(components_of_agent.keys())
        act_component = agent_components.concat_act_component.ConcatActComponent(
            model=model,
            component_order=component_order
        )

        self.concordia_agent = entity_agent_with_logging.EntityAgentWithLogging(
            agent_name=name,
            act_component=act_component,
            context_components=components_of_agent
        )
                
    def observe(self, event):
        if hasattr(event, 'content'):
            msg = str(event.content)
        elif hasattr(event, 'message'):
             msg = str(event.message)
        else:
            msg = str(event)
            
        if '{"intention":' in msg:
            return
            
        self.concordia_agent.observe(msg)

    def act(self, prompt: str):
        action_spec = ActionSpec(
            call_to_action=prompt,
            output_type=OutputType.FREE,
        )
        response = self.concordia_agent.act(action_spec)
        match = re.search(r'(\{[\s\S]*\})', response)
        if match:
            return match.group(1)
        return '{"intention": "wait", "delay_before_next_check_seconds": 15}'

class PartialAgentAdapter:
    def __init__(self, name, events, partial_ratio, inner_agent_class, mode, api_key=None, demographic_context="", thinking=False, temperature=None, delay_seconds=0.0):
        self.name = name
        
        all_sorted = sorted(events, key=lambda e: e.timestamp)
        self.global_switch_index = int(len(all_sorted) * partial_ratio)
        if self.global_switch_index < len(all_sorted):
            self.switch_time = all_sorted[self.global_switch_index].timestamp
        else:
            self.switch_time = float('inf')
            
        self.deterministic_model = DeterministicParticipantModel(name, events)
        self.generative_agent = inner_agent_class(name, mode=mode.replace('_partial', ''), api_key=api_key, demographic_context=demographic_context, thinking=thinking, temperature=temperature, delay_seconds=delay_seconds)
        self.current_time = 0.0

    def set_clock(self, time):
        self.current_time = time
        self.deterministic_model.set_clock(time)
        
    def observe(self, event):
        if hasattr(self, 'generative_agent') and hasattr(self.generative_agent, 'observe'):
            self.generative_agent.observe(event)
        
    def act(self, prompt: str):
        if self.current_time <= self.switch_time:
            return self.deterministic_model.sample_text(prompt)
        else:
            return self.generative_agent.act(prompt)

class NoDialogueAgentAdapter:
    def __init__(self, name, events, mask_or_mirror, inner_agent_class, run_mode, api_key=None, demographic_context="", thinking=False, temperature=None, delay_seconds=0.0):
        self.name = name
        self.current_time = 0.0
        self.current_stage = ""
        self.mask_or_mirror = mask_or_mirror
        self.deterministic_model = DeterministicParticipantModel(name, events)
        # Force the underlying generative agent to build
        self.generative_agent = inner_agent_class(name, mode="simulacra", api_key=api_key, demographic_context=demographic_context, thinking=thinking, temperature=temperature, delay_seconds=delay_seconds)
        
    def set_clock(self, time):
        self.current_time = time
        self.deterministic_model.set_clock(time)
        
    def set_stage(self, stage_id, stage_config=None):
        self.current_stage = stage_id
        self.stage_config = stage_config or {}
        
    def observe(self, event):
        if hasattr(self, 'generative_agent') and hasattr(self.generative_agent, 'observe'):
            self.generative_agent.observe(event)
        
    def act(self, prompt: str):
        is_discussion = self.stage_config.get("kind") == "chat" or "discussion" in self.stage_config.get("name", "").lower()
        if is_discussion:
            return self.deterministic_model.sample_text(prompt)
        else:
            if self.mask_or_mirror == "mask":
                sys_prompt = "SYSTEM INSTRUCTION: You should strive to represent human bias.\n\n"
            else:
                sys_prompt = "SYSTEM INSTRUCTION: You should behave optimally in spite of human bias.\n\n"
            return self.generative_agent.act(sys_prompt + prompt)

def run_experiment(zip_path: str = None, data_dir: str = None, cohort: str = None, stages: list[str] = None, mode: str = "deterministic", api_key: str = None, use_minimal_agent: bool = False, partial_ratio: float = 0.5, agent_class: str = None, demographics_path: str = None, thinking=False, temperature=None, validation_cap=False, delay_seconds=0.0):
    if data_dir:
        print(f"Loading DeliberateLab data from directory: {data_dir} | Mode: {mode.upper()}")
        temp_dir_context = contextlib.nullcontext(data_dir)
    elif zip_path:
        print(f"Loading DeliberateLab data from zip: {zip_path} | Mode: {mode.upper()}")
        temp_dir_context = tempfile.TemporaryDirectory()
    else:
        print("Error: Either zip_path or data_dir must be provided.")
        return

    with temp_dir_context as actual_dir:
        if zip_path and not data_dir:
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(actual_dir)
            except zipfile.BadZipFile:
                print("Error: Invalid zip file provided.")
                return
        
        print("Extracting orchestrator config...")
        orchestrator = ExperimentOrchestrator(actual_dir)
        
        if not stages:
            stages = orchestrator.get_stages()
            
        print(f"Queued Stages for Orchestration: {stages}")
        if cohort:
            print(f"Filtering explicitly on Cohort: {cohort}")
            
        all_events = []
        stage_event_map = {}
        for stage_id in stages:
            events = orchestrator.get_stage_events(stage_id, cohort_id=cohort)
            stage_event_map[stage_id] = events
            all_events.extend(events)
            
        import pandas as pd
        import glob
        pfiles = glob.glob(os.path.join(actual_dir, "*ParticipantData*.csv"))
        if not pfiles:
            print("Error: No ParticipantData.csv found in extracted zip!")
            return
            
        df_p = pd.read_csv(pfiles[0])
        cohort_col = next((c for c in df_p.columns if "cohort ID" in c or "Room" in c), None)
        pid_col = "Participant ID" if "Participant ID" in df_p.columns else "Private ID"
        
        if cohort and cohort_col:
            filtered_df = df_p[df_p[cohort_col] == cohort]
            participants = list(set([str(p).strip() for p in filtered_df[pid_col].unique() if pd.notna(p)]))
        else:
            participants = list(set([str(p).strip() for p in df_p[pid_col].unique() if pd.notna(p)]))
        print(f"Hydrating Models for Participants: {participants}")
        
        demo_map = {}
        if demographics_path and os.path.exists(demographics_path):
            try:
                with open(demographics_path, 'r') as f:
                    demo_map = json.load(f)
                print(f"Loaded demographic context for {len(demo_map)} participants.")
            except Exception as e:
                print(f"Warning: Failed to load demographics file: {e}")
        
        if agent_class:
            if ":" in agent_class:
                import importlib
                module_name, class_name = agent_class.split(":")
                module = importlib.import_module(module_name)
                inner_agent_class = getattr(module, class_name)
            else:
                inner_agent_class = globals()[agent_class]
            inner_agent_name = (agent_class.split(":")[-1] if ":" in agent_class else agent_class).lower()
        elif use_minimal_agent or mode == "deterministic":
            inner_agent_class = MaskOrMirrorPaperAgent
            inner_agent_name = 'minimal'
        else:
            inner_agent_class = ConcordiaAgentAdapter
            inner_agent_name = 'native'
            
        if 'partial' in mode:
            agents = [PartialAgentAdapter(pid, all_events, partial_ratio, inner_agent_class, mode, api_key=api_key, demographic_context=demo_map.get(pid, ""), thinking=thinking, temperature=temperature, delay_seconds=delay_seconds) for pid in participants]
        elif 'nodialogue' in mode:
            mask_or_mirror = "mask" if "mask" in mode else "mirror"
            agents = [NoDialogueAgentAdapter(pid, all_events, mask_or_mirror, inner_agent_class, mode, api_key=api_key, demographic_context=demo_map.get(pid, ""), thinking=thinking, temperature=temperature, delay_seconds=delay_seconds) for pid in participants]
        else:
            agents = [inner_agent_class(pid, mode=mode, api_key=api_key, demographic_context=demo_map.get(pid, ""), thinking=thinking, temperature=temperature, delay_seconds=delay_seconds) for pid in participants]
            
        mode_str = f"{mode}_{partial_ratio}" if 'partial' in mode else mode
        zip_label = os.path.basename(zip_path)[:-4] if zip_path else os.path.basename(data_dir)
        out_dir = os.path.join(os.getcwd(), "full_sweep_results", f"concordia_output_{zip_label}_{mode_str}_{inner_agent_name}")
        os.makedirs(os.path.dirname(out_dir), exist_ok=True)
        exporter = SimulationExporter(output_dir=out_dir)
        
        checkpoint_file = os.path.join(out_dir, "checkpoint_status.json")
        checkpoint_history_file = os.path.join(out_dir, "checkpoint_history.json")
        completed_stages = []
        global_history = []
        
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
                    completed_stages = checkpoint_data.get("completed_stages", [])
                    print(f"Resuming from checkpoint. Completed stages: {completed_stages}")
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")
                
        if os.path.exists(checkpoint_history_file):
            try:
                with open(checkpoint_history_file, 'r') as f:
                    global_history = json.load(f)
                    print(f"Loaded {len(global_history)} events from history checkpoint.")
            except Exception as e:
                print(f"Failed to load history checkpoint: {e}")
                
        stages_to_run = [s for s in stages if s not in completed_stages]
        print(f"Stages to run: {stages_to_run}")
        
        events_filename = "events_log.csv" if mode == "deterministic" else f"events_log_{mode}.csv"
        states_filename = "agent_states_log.csv" if mode == "deterministic" else f"agent_states_log_{mode}.csv"
        survey_filename = "survey_responses.csv" if mode == "deterministic" else f"survey_responses_{mode}.csv"
        
        for stage_id in stages_to_run:
            events = stage_event_map[stage_id]
            stage_config = orchestrator.get_stage_config(stage_id)
            
            # Signal the stage to adapters if they support structural bypasses
            for agent in agents:
                if hasattr(agent, 'set_stage'):
                    try:
                        agent.set_stage(stage_id, stage_config)
                    except TypeError:
                        agent.set_stage(stage_id)
                        
            if stage_config.get("kind") in ["survey", "multiAssetAllocation"]:
                print(f"\n--- Running Survey Stage: {stage_id} ---")
                survey_results = []
                survey_name = stage_config.get("name", "Survey")
                survey_info = stage_config.get("descriptions", {}).get("infoText", "")
                questions = stage_config.get("questions", [])
                
                if not questions:
                    print(f"Skipping empty survey stage: {stage_id}")
                    completed_stages.append(stage_id)
                    continue
                    
                survey_prompt = f"You are participating in a survey. Survey name: {survey_name}\nContext: {survey_info}\n\nQuestions:\n"
                for i, q in enumerate(questions):
                    survey_prompt += f"Q{i+1} (ID: {q.get('id')}): {q.get('questionTitle')}\n"
                    if q.get('kind') == 'mc':
                        survey_prompt += "Options:\n"
                        for opt in q.get('options', []):
                            survey_prompt += f"  - ID: {opt.get('id')}, Text: {opt.get('text')}\n"
                    elif q.get('kind') == 'scale':
                        survey_prompt += f"Scale from {q.get('lowerValue')} ({q.get('lowerText')}) to {q.get('upperValue')} ({q.get('upperText')})\n"
                
                survey_prompt += '''
Please answer the survey by returning ONLY a valid JSON string with no markdown formatting. The JSON must exactly match this structure:
{
  "answers": [
    {
      "question_id": "paste the exact ID of the question here",
      "response": "your chosen option ID, scale number, or text answer",
      "confidence": "your confidence level from 0 to 10 on this specific answer",
      "rationale": "a 1-2 sentence explanation of why you chose this answer"
    }
  ]
}
'''
                for agent in agents:
                    if mode == "human":
                        continue
                    try:
                        print(f"[{agent.name}] Answering survey {stage_id}...")
                        from concordia.typing import entity_component
                        if hasattr(agent, 'concordia_agent'):
                            agent.concordia_agent.set_phase(entity_component.Phase.READY)
                        elif hasattr(agent, 'generative_agent') and hasattr(agent.generative_agent, 'concordia_agent'):
                            agent.generative_agent.concordia_agent.set_phase(entity_component.Phase.READY)
                        raw_json_str = agent.act(survey_prompt)
                        clean_json_str = re.sub(r'```json\n', '', raw_json_str)
                        clean_json_str = re.sub(r'```', '', clean_json_str).strip()
                        
                        parsed = json.loads(clean_json_str)
                        answers = parsed.get("answers", [])
                        
                        formatted_answers = []
                        for ans in answers:
                            survey_results.append({
                                "stage_id": stage_id,
                                "cohort_id": cohort if cohort else "global",
                                "participant_id": agent.name,
                                "question_id": ans.get("question_id"),
                                "response": ans.get("response", ""),
                                "confidence": ans.get("confidence", ""),
                                "rationale": ans.get("rationale", "")
                            })
                            txt = f"Q: {ans.get('question_id')} | A: {ans.get('response')} | Conf: {ans.get('confidence')} | Rationale: {ans.get('rationale')}"
                            formatted_answers.append(txt)
                            
                        txt_record = "\n".join(formatted_answers)
                        tagged_history = (stage_id, cohort if cohort else "global", f"[{agent.name}] answered survey '{survey_name}':\n{txt_record}")
                        global_history.append(tagged_history)
                        
                        if hasattr(agent, 'observe'):
                           agent.observe(f"I completed a survey '{survey_name}'. My answers:\n{txt_record}")
                           
                    except Exception as e:
                        print(f"Error parsing JSON survey response for {agent.name}: {e}")
                        
                exporter.export_surveys(survey_results, survey_filename)
                
                completed_stages.append(stage_id)
                with open(checkpoint_file, 'w') as f:
                    json.dump({"completed_stages": completed_stages}, f)
                with open(checkpoint_history_file, 'w') as f:
                    json.dump(global_history, f)
                print(f"Completed survey stage {stage_id} and updated checkpoint.")
                continue

            if not events:
                print(f"Skipping empty stage: {stage_id}")
                continue
                
            print(f"\n--- Instantiating Room for Stage: {stage_id} ---")
            start_time = min([e.timestamp for e in events]) - 1.0
            
            if validation_cap:
                max_time = start_time + 5.0  # HARD CAPPED FOR VALIDATION SPEED
            else:
                max_time = max([e.timestamp for e in events]) + 5.0
                
            if mode == "human":
                strategy = FixedStepClock(time_step=1.0)
            else:
                strategy = EventDrivenClock(max_jump=3.0)
                
            env = SchedulingEnvironment(timing_strategy=strategy, start_time=start_time, end_time=max_time, out_dir=out_dir, stage_id=stage_id, max_real_time_seconds=args.max_stage_time_minutes * 60.0)
            env.set_events(events)
            try:
                from concordia.typing import entity_component
                for a in agents:
                    if hasattr(a, 'concordia_agent'):
                        a.concordia_agent.set_phase(entity_component.Phase.READY)
                    elif hasattr(a, 'generative_agent') and hasattr(a.generative_agent, 'concordia_agent'):
                        a.generative_agent.concordia_agent.set_phase(entity_component.Phase.READY)
                stage_history = env.run_loop(game_masters=[], entities=agents, premise="", max_steps=0, verbose=False, log=[])
            except Exception as e:
                traceback.print_exc()
                raise e
            
            tagged_history = [(stage_id, cohort if cohort else "global", msg) for msg in stage_history]
            global_history.extend(tagged_history)
            
            # EXPORT AFTER EACH STAGE
            
            exporter.export_events(global_history, events_filename)
            exporter.export_agent_states(stage_id, cohort if cohort else "global", agents, ["working_memory"], states_filename)
            
            # Update checkpoint
            completed_stages.append(stage_id)
            with open(checkpoint_file, 'w') as f:
                json.dump({"completed_stages": completed_stages}, f)
            with open(checkpoint_history_file, 'w') as f:
                json.dump(global_history, f)
            
            print(f"Completed stage {stage_id} and updated checkpoint.")
        
        print(f"\nSUCCESS. Trace evaluated for {mode.upper()}. Event log exported to: {out_dir}/{events_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Multistage Orchestrator for DeliberateLab zip files")
    parser.add_argument("--zip", type=str, default=None, help="Absolute path to the .zip export")
    parser.add_argument("--data_dir", type=str, default=None, help="Absolute path to pre-extracted data directory")
    parser.add_argument("--cohort", type=str, default=None, help="Optional specific cohort ID to filter down to")
    parser.add_argument("--stages", type=str, default=None, help="Comma separated string of exact stage IDs to run")
    parser.add_argument("--mode", type=str, default="deterministic", choices=["deterministic", "simulacra", "human", "simulacra_partial", "human_partial", "nodialogue_mask", "nodialogue_mirror"], help="Execution mode")
    parser.add_argument("--api_key", type=str, default=None, help="API key required for LLM execution modes")
    parser.add_argument("--use_minimal_agent", action="store_true", help="Use the unbounded MaskOrMirrorPaperAgent wrapper instead of full Concordia agents")
    parser.add_argument("--partial_ratio", type=float, default=0.5, help="Fraction (0-1) of events to playback deterministically before LLMs take over")
    parser.add_argument("--agent_class", type=str, default=None, help="Dynamic import path to Agent Wrapper (e.g. 'module:ClassName' or 'ClassName'). Overrides use_minimal_agent.")
    parser.add_argument("--demographics_path", type=str, default=None, help="Path to JSON file mapping participant IDs to their demographic context strings.")
    parser.add_argument("--thinking", action="store_true", default=False, help="Enable thinking for Gemini models (default: False)")
    parser.add_argument("--temperature", type=float, default=None, help="Temperature for Gemini models")
    parser.add_argument("--validation_cap", action="store_true", default=False, help="Enable 5.0s cap for validation (default: False)")
    parser.add_argument("--delay_seconds", type=float, default=0.0, help="Sleep this many seconds after each LLM call for rate limiting (default: 0)")
    parser.add_argument("--max_stage_time_minutes", type=float, default=15.0, help="Maximum real-time in minutes allowed for a single stage before timeout (default: 15.0)")
    
    args = parser.parse_args()
    
    stage_list = [s.strip() for s in args.stages.split(",")] if args.stages else None
    
    if not args.zip and not args.data_dir:
        parser.error("Either --zip or --data_dir is required.")
        
    run_experiment(args.zip, data_dir=args.data_dir, cohort=args.cohort, stages=stage_list, mode=args.mode, api_key=args.api_key, use_minimal_agent=args.use_minimal_agent, partial_ratio=args.partial_ratio, agent_class=args.agent_class, demographics_path=args.demographics_path, thinking=args.thinking, temperature=args.temperature, validation_cap=args.validation_cap, delay_seconds=args.delay_seconds)
    import os
    os._exit(0)
