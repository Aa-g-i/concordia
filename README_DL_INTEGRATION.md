# DeliberateLab ↔ Concordia Integration Pipeline

This pipeline enables you to seamlessly port real-world human conversation data (exported from DeliberateLab) natively into a structured Concordia simulated environment. 

By running DeliberateLab datasets through this orchestrator, you force Concordia agents (`EntityAgentWithLogging`, `long_horizon_agent`, etc.) to "observe" historically constrained dialogue step-by-step. The agents process the human inputs as if they were naturally speaking in the environment, triggering their cognitive evaluation streams (like Theory of Mind, Self Perception, and Associative Memory).

## How it works

1. Evaluates your DeliberateLab schema (`Logs.json` / `[experiment_name].json`) to establish the topology of the stages (e.g. Chat -> Survey -> Chat).
2. Uses the `ExperimentOrchestrator` to automatically build isolated Concordia `ChronoEnvironment` contexts for each physical stage.
3. Overrides the core LLM sampling behavior for specific participants (`DeterministicParticipantModel`), forcing generative text to instead explicitly trace the chronological timestamps parsed from your `ChatHistory*.csv` files.
4. Generates a flattened `events_log.csv` explicitly linking the `stage_id`, `cohort_id`, and `agent_id` for your Jupyter Notebook analysis.

## Getting Started

To execute a DeliberateLab experiment on top of Concordia, you just need a standard DeliberateLab `.zip` export file containing `*ChatHistory*.csv` records and a master JSON topology file. 

You can run the script via the lightweight launcher:

```bash
cd INSTALLS/concordia
./launch_dl.sh <path_to_zip> [cohort_id] [stages]
```

### Examples

**1. Run an entire experiment (e.g. A5) start to finish exactly as it happened:**
```bash
./launch_dl.sh ~/Downloads/a5_experiment_data.zip
```
> This will execute sequentially through every stage defined in the `.json` `stageMap`.

**2. Focus on a specific Breakout Room (Cohort):**
```bash
./launch_dl.sh ~/Downloads/a5_experiment_data.zip "mixk3s38-ey2c1o"
```
> The Orchestrator will slice the data, applying strict visibility so the simulation only instantiates models for the humans explicitly assigned to `mixk3s38-ey2c1o`.

**3. Test Specific Stages Temporally (Isolation filtering):**
```bash
./launch_dl.sh ~/Downloads/a5.zip "mixk3s38" "discussion-round-1,discussion-round-3"
```
> Skips `round-2` and forces the simulation to bridge agent memory linearly between Stage 1 and Stage 3.

## Generated Output

Upon success, you will find a generated directory (`concordia_output_[zipname]`) containing universally relational log dumps:

*   **`events_log.csv`**: Contains every single event explicitly mapping `stage_id`, `cohort_id`, `timestamp`, `agent_id`, and `content`.
*   **`agent_states_log.csv`**: Contains scraped values of the underlying component memories (`TheoryOfMind`, `WorkingMemory`, etc) associated with the agents globally.
