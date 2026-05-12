"""Parses DeliberateLab exported data (CSV/JSON) into structures for Concordia."""

import csv
import dataclasses
from datetime import datetime
from collections.abc import Sequence

@dataclasses.dataclass
class ChatEvent:
    timestamp: float
    participant_id: str
    message: str

@dataclasses.dataclass
class SurveyEvent:
    timestamp: float
    participant_id: str
    survey_prompt: str
    survey_response: str
    
def load_chat_logs(csv_path: str) -> list[ChatEvent]:
    """Loads a DL ChatLog CSV into a sequence of ChatEvents."""
    events = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        # Determine actual column names since DL exports can vary slightly
        headers = reader.fieldnames or []
        time_col = next((c for c in headers if 'timestamp' in c.lower()), 'timestamp')
        pid_col = next((c for c in headers if c.lower() in ['sender id', 'participant_id', 'participant id', 'participant public id']), 'participant_id')
        msg_col = next((c for c in headers if c.lower() in ['message content', 'message', 'text']), 'message')
        
        for row in reader:
            try:
                # Need to convert DL timestamp (e.g. "2025-11-18 00:05:19.000") to float
                t_str = row[time_col]
                try:
                    dt = datetime.strptime(t_str, "%Y-%m-%d %H:%M:%S.%f")
                    t_val = dt.timestamp()
                except ValueError:
                    t_val = float(t_str) # Fallback to already float
                
                # Only log non-empty participant messages
                msg = row[msg_col].strip()
                if msg:
                    events.append(ChatEvent(
                        timestamp=t_val,
                        participant_id=row[pid_col],
                        message=msg
                    ))
            except (KeyError, ValueError) as e:
                pass # skip invalid rows or system messages missing some fields
                
    # Ensure they are sorted temporally to guarantee correct trace playback
    return sorted(events, key=lambda e: e.timestamp)
