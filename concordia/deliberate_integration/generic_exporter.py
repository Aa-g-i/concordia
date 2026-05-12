"""Exports generalized Concordia traces to relational CSV files."""

import csv
import os
from typing import Sequence, Any

class SimulationExporter:
    """Exports events and agent states from a Concordia run."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def export_events(self, chat_history: Sequence[Any], filename: str = "events_log.csv"):
        """Exports the chronological chat event log."""
        out_path = os.path.join(self.output_dir, filename)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["stage_id", "cohort_id", "time_index", "agent_id", "content"])
            
            for idx, item in enumerate(chat_history):
                # If tuple, unpack the metadata
                if isinstance(item, tuple) and len(item) == 3:
                    stage_id, cohort_id, event = item
                else:
                    stage_id, cohort_id, event = "unknown", "unknown", str(item)
                    
                # Attempt to parse "[001.0] Alice: Hello!"
                if event.startswith("[") and ":" in event:
                    time_end = event.find("]")
                    time_str = event[1:time_end]
                    remainder = event[time_end+1:].strip()
                    if ":" in remainder:
                        agent_id, content = remainder.split(":", 1)
                        writer.writerow([stage_id, cohort_id, time_str, agent_id.strip(), content.strip()])
                    else:
                        writer.writerow([stage_id, cohort_id, time_str, "System", remainder.strip()])
                else:
                    writer.writerow([stage_id, cohort_id, idx, "System", event])

    def export_agent_states(self, stage_id: str, cohort_id: str, agents: Sequence[Any], metrics: Sequence[str], filename: str = "agent_states_log.csv"):
        """Exports requested internal component states of the agents mapped to the current stage context."""
        out_path = os.path.join(self.output_dir, filename)
        
        # Append to explicit file to preserve chronological stack across stages
        write_mode = "a" if os.path.exists(out_path) else "w"
        
        with open(out_path, write_mode, newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_mode == "w":
                writer.writerow(["stage_id", "cohort_id", "agent_id", "metric_name", "value_final"])
            
            for agent in agents:
                # Mock extraction: if agents were real concordia Entity objects,
                # we would call agent.state() to get the component dictionary.
                # For `long_horizon_agent` it might contain 'working_memory' or 'self_perception'
                comp_states = {}
                if hasattr(agent, "state"):
                    try:
                        comp_states = agent.state()
                    except Exception:
                        pass
                
                for metric in metrics:
                    val = comp_states.get(metric, "N/A")
                    writer.writerow([stage_id, cohort_id, agent.name, metric, val])
                    
    def export_surveys(self, survey_results: list[dict], filename: str = "survey_responses.csv"):
        """Exports parsed JSON survey responses."""
        if not survey_results:
            return
            
        out_path = os.path.join(self.output_dir, filename)
        file_exists = os.path.exists(out_path)
        
        with open(out_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["stage_id", "cohort_id", "participant_id", "question_id", "response", "confidence", "rationale"])
            if not file_exists:
                writer.writeheader()
            for row in survey_results:
                writer.writerow(row)
                
    def append_to_manifest(self, filepath: str, manifest_name: str = "file_list.txt"):
        """Appends the generated file path to a manifest for notebook ingestion."""
        manifest_path = os.path.join(self.output_dir, manifest_name)
        with open(manifest_path, "a", encoding="utf-8") as f:
            f.write(f"{filepath}\n")
