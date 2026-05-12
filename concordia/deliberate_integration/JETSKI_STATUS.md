# Jetski Workspace Status: Deliberate Integration & Simulation Tracking

**Last Updated:** 2026-04-22

## Context
We are surveying the Concordia + DeliberateLab integration simulations to identify missing runs and complete the analysis pipeline. We also just completed the migration of the legacy simulation strategies to the core `SchedulingEnvironment` architecture.

## Vision
The overarching goal of this project is to evaluate and benchmark the capability of Large Language Model (LLM) agents (using the Concordia framework) to simulate human deliberation and behavior (from DeliberateLab experiments). By comparing simulation artifacts (dialogue, survey responses, internal states) against real human baselines **and across different simulation conditions**, we aim to:
- **Measure Behavioral Fidelity**: Quantify how closely LLM agents mirror or mask human biases.
- **Analyze Methodological Deltas**: Compare the performance and artifacts of different simulation modes (e.g., 'Simulacra' vs 'Human' mode, full vs partial replay, Native vs Minimal agents).
- **Comprehensive Metrics Framework**: We track 10 core metrics to characterize the simulations, implemented in [analysis_comprehensive_metrics.py](file:///usr/local/google/home/aarontp/INSTALLS/concordia/concordia/deliberate_integration/analysis/analysis_comprehensive_metrics.py). This is our preferred analysis style as it provides a complete picture of the simulation quality.
    - **Survey**: KL Divergence, MAE.
    - **Dialogue**: KL Divergence (Verbosity), HDS (Verbosity Deviation), Gini (Inequality), Timing, TTR (Lexical Diversity), Transition Entropy, Semantic Convergence.
    - **Other**: Proxy KL.

The **artifacts and notebooks** are the instruments for this analysis, with `analysis_comprehensive_metrics.py` being the primary tool for generating the comparative plots and summary tables.

## Current State
- **Sweep Jobs Total:** 201 (from `sweep_jobs.txt`)
- **Logs Found:** 201 in `sweep_logs/`
- **Status:** All 201 jobs listed in `sweep_jobs.txt` have corresponding logs with a "SUCCESS" marker.

## Treatments Matrix

We are tracking the following combinations of conditions and treatments across datasets (e.g., `f1_200`, `f1a_200`, `f2a_300`).

### Agent Architecture Treatments
*   **Native**: Full Concordia `MemoryAgent` with associative memory (from the Concordia framework).
*   **Minimal**: A simpler prompting scheme that just concatenates history (based on the "Mask vs Mirror" paper).

### Matrix

| Mode (Prompting) | Agent Type (Architecture) | Control Mode | Status |
| :--- | :--- | :--- | :--- |
| **Human** (Mirror) | Minimal | Full Simulation | Completed |
| **Human** (Mirror) | Native | Full Simulation | Completed |
| **Simulacra** (Mask) | Minimal | Full Simulation | Completed |
| **Simulacra** (Mask) | Native | Full Simulation | Completed |
| **Simulacra** (Mask) | Minimal | Partial (0.5 Replay) | Completed |
| **Simulacra** (Mask) | Native | Partial (0.5 Replay) | Completed |
| **Mask** | Native | NoDialogue (Playback) | Completed |
| **Mirror** | Native | NoDialogue (Playback) | Completed |

*(Note: The folder names use `mask` and `mirror` for NoDialogue modes, which map to Simulacra and Human behaviors respectively).*

## Consolidated Nomenclature (Approved 2026-04-30)

To avoid confusion with legacy terms like "Mask" and "Mirror", we use the following standardized names in reports and plots:

| Legacy/Folder Pattern | Consolidated Name | Description |
| :--- | :--- | :--- |
| `simulacra_minimal` | **Minimal-Simulacra** | Minimal agent, full simulation in Simulacra mode |
| `simulacra_partial_0.5_minimal` | **Minimal-Simulacra-Partial** | Minimal agent, 0.5 replay in Simulacra mode |
| `simulacra_native` | **Native-Simulacra** | Native agent, full simulation in Simulacra mode |
| `simulacra_partial_0.5_native` | **Native-Simulacra-Partial** | Native agent, 0.5 replay in Simulacra mode |
| `nodialogue_mask_native` | **Playback-Simulacra** | Native agent, playing back Simulacra dialogue |
| `nodialogue_mirror_native` | **Playback-Human** | Native agent, playing back Human dialogue |
| `human_minimal` | **Human** | The human baseline data (simulated dialogue) |

## Architecture & Migration Status
- **Environment:** Migrated from legacy `ChronoEnvironment` to core `SchedulingEnvironment`.
- **Clock Strategy:** Implemented `EventDrivenClock` with event lock-on and a 3-second silence cap (`max_jump`) to simulate realistic human dialogue pacing.
- **Tests:** Stubbed out legacy `chrono_environment_test.py` and `chrono_playback_test.py` because they relied on the deleted `ChronoEnvironment`. All other 332 core tests and new environmental clock tests passed successfully.

## Updates & Reporting Synthesis
- **Script Implementation:** Refactoring of `collate_sweep_results.py` completed. Directories and targets are now flexible and dynamically configurable via flags `--base-dir` and `--output-md`.
- **Comprehensive Synthesis:** Generated `comprehensive_analysis_report.md` (previous session), `sweep_results.md`, `full_inventory_and_comparison_report.md`, and `narrative_report.md` overviewing data aggregation comparisons, UMAP plots, and narrative synthesis.
- **Tests Restored:** Created `playback_test.py` to restore valid assertions from `chrono_playback_test.py` for the new `SchedulingEnvironment`.
- **Branch Management:** Created `feature/deliberate-review` branch excluding deferred agent enhancements, ready for review.

## Missing / Incomplete Simulations & Metrics
- None found based on `sweep_jobs.txt`. All listed jobs appear to have completed successfully.
- **Proxy KL**: Confirmed that `Native-Simulacra-Full` and `Native-Simulacra-Partial` are missing from `proxy_kl_experiment.csv` (likely never run for these models).

## Analysis Todos (from Session 2026-04-30)
1. **Fix Human Baseline Mapping**: Map participants to cohorts in human dialogue data (currently all "global") using `survey_responses.csv` to allow valid cohort-level comparisons (Gini, TTR, etc.).
2. **Investigate Missing Metrics**: Determine why `Dialogue KL` is missing for all `Native` methods in `comprehensive_metrics_comparison.csv`.
3. **Align Naming Conventions**: Standardize method names across `analysis_comprehensive_metrics.py` and `analysis_global_metrics.py` (e.g., `LLM Simulacra` vs `Minimal-Simulacra-Full`).

## Next Steps for Jetski
1. **Address Analysis Todos**: Complete the items above to ensure coherent and shareable results.
2. **Run new simulations**: Verify that the simulation pipeline yields valid results on new data.
3. **Update Documentation**: Reference this status file in `README.md`.
4. **Push for Review**: Push `feature/deliberate-review` and create a PR.

