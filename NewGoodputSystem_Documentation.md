# New Goodput Measurement System Documentation

## 1. Overview

The new goodput measurement system is powered by the `ml-goodput-measurement` Python library. It offers a more granular, standardized, and detailed approach to tracking and analyzing the efficiency of your training jobs compared to the previous system.

**Benefits:**
*   **Granular Event Tracking:** Captures distinct phases of the job lifecycle (e.g., TPU initialization, data loading, training prep) to pinpoint inefficiencies.
*   **Standardized Metrics:** Provides consistent definitions for goodput and various badput categories.
*   **Actionable Insights:** Detailed badput breakdown helps identify specific areas for optimization.
*   **Extensibility:** Allows logging of custom badput events relevant to your specific workflows.

This system revolves around two main components:
*   `GoodputRecorder`: Integrated into your training script to log key events.
*   `GoodputCalculator` (and its CLI wrapper `new_goodput_calculator_cli.py`): Used to process these events and calculate goodput/badput metrics.

## 2. Logging Events with `GoodputRecorder`

The `GoodputRecorder` class from the `ml-goodput-measurement` library is responsible for capturing timestamps and metadata for important events during your training job's lifecycle. Proper instrumentation of your training script with `GoodputRecorder` is crucial for accurate metrics.

**For detailed instructions on how to integrate `GoodputRecorder` into your training scripts, please refer to the `GoodputRecorder_Integration_Guide.md` document.**

In summary, `GoodputRecorder` should be used to log events such as:
*   **`user_scheduled`**: When the user or an automated system schedules the job.
*   **`job_started`**: When the job's main process begins execution on the allocated resources.
*   **`tpu_init_started` / `tpu_init_completed`**: Marks the start and end of TPU system initialization.
*   **`training_prep_started` / `training_prep_completed`**: Marks the start and end of training preparation activities (e.g., model compilation, initial data pipeline setup).
*   **`data_loading_started` / `data_loading_completed`**: Records the time taken to load data for each training step.
*   **`step_started` / `step_completed`**: Times the duration of each individual training step (forward/backward pass, optimizer step).
*   **`checkpoint_saved` / `checkpoint_loaded`**: Logs checkpoint operations.
*   **`job_terminated`**: When the job finishes (successfully, with failure, or preempted).
*   **`custom_badput_event`**: Allows logging of any other specific inefficiencies or delays relevant to your job.

## 3. Calculating and Viewing Metrics

Once events are logged by `GoodputRecorder` and stored in the backend (e.g., BigQuery), you can calculate and view goodput metrics.

### Using `new_goodput_calculator_cli.py`

The `new_goodput_calculator_cli.py` script provides a command-line interface to the `GoodputCalculator`.

**Basic Usage:**
```bash
python new_goodput_calculator_cli.py --job_name "your_job_name_here"
```
*   `--job_name`: (Required) The name of the job you want to analyze. The script typically derives a `logger_name` (e.g., `goodput_your_job_name_here`) from this to fetch the events.
*   `--logger_name`: (Optional) If your events were logged with a specific `logger_name` that doesn't follow the `goodput_<job_name>` pattern, you can specify it directly.

The script will output:
*   Overall Goodput Percentage.
*   A breakdown of Badput sources by percentage.
*   Key job timeline information (start/end times, last step).

### Using `GoodputCalculator` Directly in Python

For more advanced or custom analyses, you can use the `GoodputCalculator` class directly in your Python scripts:

```python
from ml_goodput_measurement.goodput import GoodputCalculator

# The logger_name should match what was used by GoodputRecorder
calculator = GoodputCalculator(logger_name="goodput_your_job_name_here")

# Fetch and calculate goodput
# include_badput_breakdown=True provides detailed badput sources
goodput_data = calculator.get_job_goodput(include_badput_breakdown=True)

if goodput_data:
    print(f"Goodput Percentage: {goodput_data.goodput_percent:.2f}%")
    # Access other attributes like goodput_data.badput_breakdown_percent, etc.
    # Refer to the GoodputResult data class definition for all available fields.
```
This allows for programmatic access to all calculated metrics and event data for deeper analysis or integration into other reporting tools.

## 4. Interpreting Metrics

### Goodput Percentage

**Goodput Percentage** = (`Total Goodput Time` / `Total Measured Duration`) * 100

*   **Total Goodput Time:** The sum of durations spent in actual productive training work, primarily the time recorded by `step_completed` events (i.e., time spent in the training step itself).
*   **Total Measured Duration:** The total wall-clock time from `job_started` to `job_terminated`.
*   **Total Badput Time:** `Total Measured Duration` - `Total Goodput Time`.

A higher goodput percentage indicates a more efficient job with less time spent on overhead or unproductive activities.

### Common Badput Types

The `GoodputCalculator` categorizes non-goodput time (Badput) into several types. Understanding these can help you pinpoint areas for optimization:

*   **`PROGRAM_STARTUP`**:
    *   **Definition:** Time from `job_started` until the first significant event occurs (e.g., `tpu_init_started`, `training_prep_started`, or the first `step_started` if earlier events are missing).
    *   **Insights:** High `PROGRAM_STARTUP` time might indicate delays in your script's initial setup, resource contention before specific initializations begin, or slow Python interpreter startup/imports.

*   **`TPU_INITIALIZATION`** (or similar for other hardware, e.g., `GPU_INITIALIZATION`):
    *   **Definition:** Time spent initializing the accelerator hardware, typically measured between `tpu_init_started` and `tpu_init_completed` events.
    *   **Insights:** Long TPU initialization times can be a significant source of badput, especially for shorter jobs. This could be due to system-level issues, complex mesh configurations, or the need for compiler caching.

*   **`TRAINING_PREP`**:
    *   **Definition:** Time spent on preparations after hardware initialization but before the training loop starts. This is typically measured between `training_prep_started` and `training_prep_completed`. It can include model compilation, initial data pipeline setup, loading pre-trained weights (if not part of checkpoint loading), etc.
    *   **Insights:** If this is high, consider optimizing model compilation (e.g., caching XLA compilations), or streamlining any one-off setup tasks.

*   **`DATA_LOADING_SYNC`** (or simply `DATA_LOADING`):
    *   **Definition:** Time spent explicitly waiting for data to be available for a training step. This is measured by the duration between `data_loading_started` and `data_loading_completed` events *outside* the main training step time.
    *   **Insights:** Significant time here indicates an I/O bottleneck or an inefficient data pipeline. Prefetching, parallelizing data loading, or optimizing data transformations are common solutions.

*   **`CHECKPOINT_OPERATIONS`** (may be broken down further, e.g. `CHECKPOINT_SAVE`, `CHECKPOINT_LOAD`):
    *   **Definition:** Time spent saving or loading checkpoints.
    *   **Insights:** Frequent or slow checkpointing can consume considerable time. Consider optimizing checkpointing frequency, using asynchronous checkpointing, or improving storage speed.

*   **`WASTED_PROGRESS_FROM_DISRUPTION`**:
    *   **Definition:** Time spent on training steps that did not contribute to the final progress due to a preemption or job failure before a checkpoint could be saved. It's the goodput from the last successful checkpoint until the job termination, if the job didn't complete normally.
    *   **Insights:** High values here highlight the cost of preemptions or crashes. More frequent checkpointing can reduce this, but at the cost of increased `CHECKPOINT_SAVE` time.

*   **`CUSTOM_BADPUT_EVENTS`**:
    *   **Definition:** Time explicitly logged using `recorder.custom_badput_event()`. This category aggregates all time attributed to custom-defined badput periods.
    *   **Insights:** This depends on what custom events you define. It could be anything from known system slowdowns, retry attempts, to specific non-training activities you want to track as badput.

*   **`OTHER`** (or `UNACCOUNTED_BADPUT`):
    *   **Definition:** Any time within the `job_started` to `job_terminated` window that is not classified as goodput or any of the other specific badput categories. This often includes time between training steps, time between the end of the last step and `job_terminated`, or gaps not covered by specific `GoodputRecorder` events.
    *   **Insights:** A large amount of "OTHER" badput suggests that the current event logging might be missing some significant non-training activities. You might need to add more specific `GoodputRecorder` calls or custom badput events to categorize this time better. It can also include Python overhead, brief sleeps, or minor computations not covered by other events.

## 5. Comparison to Old System

The new `ml-goodput-measurement` system provides a more detailed and structured view of job performance than the old system. While both aim to quantify "goodput," their methodologies and the granularity of badput categorization differ.

*   **Direct Percentage Comparison:** Directly comparing goodput percentages between the old and new systems might be misleading due to these differences.
*   **Granular Insights:** The new system's strength lies in its detailed badput breakdown. This allows for more precise identification of bottlenecks that the old system might have obscured or aggregated differently.

Focus on leveraging the new system's detailed breakdown for optimization efforts, rather than striving for a 1:1 match with historical metrics from the old system.

## 6. Troubleshooting/Key Considerations

*   **Library Installation:** Ensure the `ml-goodput-measurement` library is correctly installed in your training environment and accessible via `PYTHONPATH`.
*   **Correct `GoodputRecorder` Integration:** Refer to `GoodputRecorder_Integration_Guide.md`. Incorrect or incomplete instrumentation will lead to inaccurate metrics. Pay close attention to event placement and `run_key`/`logger_name` consistency.
*   **Event Data Routing:** Verify that the events logged by `GoodputRecorder` are being correctly routed to and stored in the backend data store (e.g., BigQuery, log files) that `GoodputCalculator` is configured to read from.
*   **`logger_name` Consistency:** The `logger_name` used when instantiating `GoodputCalculator` (or passed via `new_goodput_calculator_cli.py`) must match the `logger_name` used by `GoodputRecorder` during the training job.
*   **Clock Synchronization:** If your job runs across multiple machines, ensure their clocks are reasonably synchronized for accurate duration measurements.
*   **Interpreting `OTHER` Badput:** If `OTHER` badput is consistently high, review your `GoodputRecorder` integration to see if additional events can capture unclassified time periods.

---
