"""Experiment Orchestrator for linking multiple DeliberateLab stages."""

import os
import json
import glob
from concordia.utils.deliberate_parser import load_chat_logs


class ExperimentOrchestrator:
    def __init__(self, extracted_dir: str):
        self.extracted_dir = extracted_dir
        self.config = self._load_config()
        self.stages = self._extract_stage_sequence()

    def _load_config(self):
        json_files = glob.glob(os.path.join(self.extracted_dir, "*.json"))
        # Prioritize the main config over Logs.json
        main_config = [f for f in json_files if "Logs" not in f]
        if not main_config:
            raise FileNotFoundError("Could not locate main Experiment config JSON.")
        with open(main_config[0], 'r', encoding='utf-8', errors='replace') as f:
            return json.load(f)

    def _extract_stage_sequence(self) -> list[dict]:
        """Extracts the topologically sorted stages from the JSON config."""
        stage_map = self.config.get("stageMap", {})
        
        # DeliberateLab stores the explicit chronological order in experiment -> stageIds
        stage_ids = self.config.get("experiment", {}).get("stageIds", [])
        
        # Fallback to stage_map keys if stageIds is missing
        if not stage_ids:
             stage_ids = list(stage_map.keys())

        seq = []
        for stage_id in stage_ids:
             data = stage_map.get(stage_id, {})
             seq.append({
                 "id": stage_id,
                 "kind": data.get("kind", "unknown"),
                 "config": data
             })
        return seq

    def get_stage_events(self, stage_id: str, cohort_id: str = None) -> list:
        """Dynamically locate and parse ChatHistory or Surveys for a given Stage ID."""
        stage_info = next((s for s in self.stages if s["id"] == stage_id), None)
        if not stage_info:
            return []

        if stage_info["kind"] == "chat":
            # Find the ChatHistory CSV for this stage
            pattern = os.path.join(self.extracted_dir, f"*ChatHistory*_Stage-{stage_id}.csv")
            matches = glob.glob(pattern)
            
            # If a specific cohort is requested, filter matches:
            if cohort_id:
                matches = [m for m in matches if cohort_id in m]
                
            if not matches:
                return []
                
            # Parse the matched files
            all_stage_events = []
            for m in matches:
                all_stage_events.extend(load_chat_logs(m))
            return all_stage_events
            
        elif stage_info["kind"] in ["survey", "multiAssetAllocation"]:
            # TODO: We can parse ParticipantData.csv mapping columns ending with '- Survey {stage_id}' 
            # into SurveyEvents to inject into the environment if needed.
            return []
            
        return []

    def get_stages(self) -> list[str]:
        return [s["id"] for s in self.stages]
        
    def get_stage_config(self, stage_id: str) -> dict:
        """Returns the raw JSON config dictionary for a given stage."""
        stage_info = next((s for s in self.stages if s["id"] == stage_id), None)
        return stage_info.get("config", {}) if stage_info else {}
