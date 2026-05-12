"""Contains the DeterministicParticipantModel that ignores prompts and plays back human data for a sanity check."""

import json
from typing import Collection, Sequence

from concordia.language_model import language_model
from concordia.utils.deliberate_parser import ChatEvent, SurveyEvent


class DeterministicParticipantModel(language_model.LanguageModel):
    """
    A Level-1 mock model that ignores Concordia's GameMaster prompts
    for conversation, but evaluates if the prompt is asking for a survey,
    and outputs exact, deterministic JSON intention structures based on human logs.
    """
    def __init__(self, participant_id: str, events: Sequence[ChatEvent | SurveyEvent]):
        self._participant_id = participant_id
        # Filter for only this person's historical events
        self._events = [e for e in events if e.participant_id == participant_id]
        self._event_index = 0
        self._clock_time = 0.0  # Kept updated by GameMaster

    def set_clock(self, time: float):
        self._clock_time = time

    def sample_text(
        self,
        prompt: str,
        *,
        max_tokens: int = 1000,
        max_characters: int = 10000,
        terminators: Collection[str] = (),
        temperature: float = 0.5,
        timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
        seed: int | None = None,
    ) -> str:
        """Evaluates whether to speak, wait, or answer a survey based on the prompt and clock."""

        # 1. Survey Fallback Check
        # If the GameMaster halts time to ask a question (e.g., ranking)
        task_prompt = prompt.split("Task:\n")[-1].lower() if "Task:\n" in prompt else prompt.lower()
        if "rank" in task_prompt or "survey" in task_prompt or "score" in task_prompt:
            # Find the next SurveyEvent (ignoring clock for private offline prompts)
            for i in range(self._event_index, len(self._events)):
                if isinstance(self._events[i], SurveyEvent):
                    self._event_index = i + 1
                    return self._events[i].survey_response
            return "[]" # Default empty survey response


        # 2. Chronological Chat check
        if self._event_index < len(self._events):
            next_event = self._events[self._event_index]

            # If the next event is not a ChatEvent (e.g. they skipped speaking until a survey), keep waiting
            if not isinstance(next_event, ChatEvent):
                return json.dumps({"action": "wait", "delay": 15.0})

            # If the clock hasn't reached the human's exact timestamp, output WAIT
            if next_event.timestamp > self._clock_time:
                return json.dumps({"action": "wait", "delay": next_event.timestamp - self._clock_time})

            # The clock has reached or passed the timestamp! Emit SPEAK intention.
            self._event_index += 1
            return json.dumps({"action": "speak", "speech": next_event.message})

        return json.dumps({"action": "wait", "delay": 15.0}) # Human did not do anything else

    def sample_choice(
        self,
        prompt: str,
        responses: Sequence[str],
        *,
        seed: int | None = None,
    ) -> tuple[int, str, dict[str, float]]:
        """Returns the first choice for generic multiple-choice queries."""
        return 0, responses[0], {r: 1.0 if i == 0 else 0.0 for i, r in enumerate(responses)}
