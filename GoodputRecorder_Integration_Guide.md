# Integration Guide: ml_goodput_measurement.GoodputRecorder

This guide provides instructions for integrating the new `GoodputRecorder` from the `ml-goodput-measurement` library into your training scripts, replacing the old `GoodputLogger`.

## 1. Instantiating GoodputRecorder

To begin using the `GoodputRecorder`, you need to import it and create an instance. It's recommended to do this early in your training script.

```python
from ml_goodput_measurement import GoodputRecorder
from ml_goodput_measurement import Event # For custom events

# Instantiate the recorder
# You might need to pass configuration parameters depending on the implementation
# For example, a run_key or experiment_id
recorder = GoodputRecorder(run_key="my_training_run_001")
```

## 2. Mapping from GoodputLogger to GoodputRecorder

The `GoodputRecorder` offers a more structured way to log events compared to the old `GoodputLogger`. Here's how the previous events map to the new methods:

*   **`USER_SCHEDULED`**: This event typically marks the very beginning of the user's intent to run a job.
    *   **Old:** `GoodputLogger.log_event(EventType.USER_SCHEDULED, timestamp_ms=...)`
    *   **New:** `recorder.user_scheduled(timestamp_ms=...)`
        *   *Placement:* At the earliest point when the user's request to start training is registered.

*   **`JOB_STARTED`**: This event signifies that the training job has actually started processing on the allocated resources.
    *   **Old:** `GoodputLogger.log_event(EventType.JOB_STARTED, timestamp_ms=...)`
    *   **New:** `recorder.job_started(timestamp_ms=...)`
        *   *Placement:* After resource allocation and just before the training script begins its core logic.

*   **`CHECKPOINT_LOADED`**: Indicates that a checkpoint has been successfully loaded.
    *   **Old:** `GoodputLogger.log_event(EventType.CHECKPOINT_LOADED, timestamp_ms=..., metadata={'path': '...'})`
    *   **New:** `recorder.checkpoint_loaded(timestamp_ms=..., path='...')`
        *   *Placement:* Immediately after a checkpoint loading operation completes.

*   **`CHECKPOINT_SAVED`**: Indicates that a checkpoint has been successfully saved.
    *   **Old:** `GoodputLogger.log_event(EventType.CHECKPOINT_SAVED, timestamp_ms=..., metadata={'path': '...'})`
    *   **New:** `recorder.checkpoint_saved(timestamp_ms=..., path='...')`
        *   *Placement:* Immediately after a checkpoint saving operation completes.

*   **`JOB_TERMINATED`**: Marks the end of the training job, whether due to completion, error, or preemption.
    *   **Old:** `GoodputLogger.log_event(EventType.JOB_TERMINATED, timestamp_ms=...)`
    *   **New:** `recorder.job_terminated(timestamp_ms=..., status='COMPLETED'|'FAILED'|'PREEMPTED', details='...')`
        *   *Placement:* At the very end of the training script, or in `finally` blocks to ensure it's called even if errors occur. The `status` and `details` parameters provide more context.

*   **`USER_TERMINATED`**: This event is similar to `JOB_TERMINATED` but might be used if the termination is explicitly initiated by the user before the job naturally concludes.
    *   **Old:** `GoodputLogger.log_event(EventType.USER_TERMINATED, timestamp_ms=...)`
    *   **New:** `recorder.user_terminated(timestamp_ms=...)` (or potentially reuse `job_terminated` with a specific status like 'USER_CANCELLED')
        *   *Placement:* When a user cancellation request is processed.

## 3. New GoodputRecorder Methods and Their Placement

`GoodputRecorder` introduces new methods to capture more granular phases of the training lifecycle, helping to pinpoint inefficiencies.

*   **`tpu_init_started(timestamp_ms=...)` and `tpu_init_completed(timestamp_ms=...)`**
    *   **Purpose:** Measures the time taken to initialize TPU resources.
    *   **Placement:**
        *   `tpu_init_started`: Just before initiating TPU system initialization.
        *   `tpu_init_completed`: Immediately after TPU initialization is confirmed.
    ```python
    # Conceptual example
    recorder.tpu_init_started()
    initialize_tpu_system() # Your TPU initialization function
    recorder.tpu_init_completed()
    ```

*   **`training_prep_started(timestamp_ms=...)` and `training_prep_completed(timestamp_ms=...)`**
    *   **Purpose:** Measures the time for overall training preparation, which might include model compilation, initial data pipeline setup (but not active loading for the first step).
    *   **Placement:**
        *   `training_prep_started`: After TPU/resource initialization, before model compilation or other setup.
        *   `training_prep_completed`: After all general setup is done, just before starting the training loop.
    ```python
    # Conceptual example
    recorder.training_prep_started()
    compile_model()
    prepare_initial_data_structures()
    recorder.training_prep_completed()
    ```

*   **`data_loading_started(timestamp_ms=..., step_number=...)` and `data_loading_completed(timestamp_ms=..., step_number=..., items_loaded=...)`**
    *   **Purpose:** Measures the time taken to load data for a specific training step.
    *   **Placement:** Inside the training loop, bracketing the data loading call for each step.
    ```python
    # Conceptual example (inside training loop)
    for step in range(num_steps):
        recorder.data_loading_started(step_number=step)
        batch = load_data_for_step(step) # Your data loading function
        recorder.data_loading_completed(step_number=step, items_loaded=len(batch))
        # ... rest of the training step
    ```

*   **`step_started(timestamp_ms=..., step_number=...)` and `step_completed(timestamp_ms=..., step_number=..., metrics={...})`**
    *   **Purpose:** Measures the duration of individual training steps (forward pass, backward pass, optimizer step).
    *   **Placement:** Inside the training loop, bracketing the core training operations.
    ```python
    # Conceptual example (inside training loop)
    for step in range(num_steps):
        # ... data loading ...
        recorder.step_started(step_number=step)
        loss = train_step_function(batch) # Your core training function
        recorder.step_completed(step_number=step, metrics={'loss': loss.item()})
        # ... checkpointing, logging, etc. ...
    ```

*   **`custom_badput_event(event_name: str, timestamp_ms=..., details='...', event_type: Event = Event.BADPUT, severity: str = 'WARNING')`**
    *   **Purpose:** Allows logging of custom events that represent "badput" or inefficiencies, such as system slowdowns, retries, or unexpected delays not covered by standard events.
    *   **Placement:** Wherever a known type of inefficiency or unexpected behavior occurs.
    ```python
    # Conceptual example
    try:
        # Some operation that might fail and be retried
        perform_critical_operation()
    except Exception as e:
        recorder.custom_badput_event(
            event_name="CriticalOperationRetry",
            details=str(e),
            severity="ERROR"
        )
        # ... retry logic ...
    ```

## 4. Conceptual Python Code Examples

Here's a more holistic (though still conceptual) view of how these events might be placed in a typical training script:

```python
from ml_goodput_measurement import GoodputRecorder, Event
import time

# Define custom exceptions for clarity if not already available
class UserCancellationException(Exception):
    pass

class PreemptionException(Exception):
    pass

# Placeholder functions for training operations
def get_user_schedule_time_from_environment(): return int(time.time() * 1000) - 10000
def initialize_tpu_system(): pass
def create_model(): return None
def create_optimizer(): return None
def compile_model_for_tpu(model): return model
def should_load_checkpoint(): return False
def load_checkpoint(model, path): pass
def get_next_batch(iterator): return [1, 2, 3] # Dummy batch
def training_step(model, optimizer, batch): return 0.5 # Dummy loss
def save_checkpoint(model, path): pass
def check_for_system_issues(): return False # Dummy check
class DatasetIterator: # Dummy iterator
    def __init__(self): self.count = 0
    def __next__(self): self.count += 1; return [self.count] 
dataset_iterator = DatasetIterator()


def main_training_loop():
    # --- Early script execution ---
    user_schedule_time = get_user_schedule_time_from_environment() # Placeholder
    recorder = GoodputRecorder(run_key="my_training_run_001")
    recorder.user_scheduled(timestamp_ms=user_schedule_time)

    try:
        # --- Job Initialization ---
        recorder.job_started(timestamp_ms=int(time.time() * 1000))

        # --- TPU/Resource Initialization ---
        recorder.tpu_init_started()
        initialize_tpu_system() # Your TPU init code
        recorder.tpu_init_completed()

        # --- Training Preparation (model compilation, etc.) ---
        recorder.training_prep_started()
        model = create_model()
        optimizer = create_optimizer()
        compiled_model = compile_model_for_tpu(model) # Your model compilation
        recorder.training_prep_completed()

        # --- Optional: Load Checkpoint ---
        if should_load_checkpoint():
            checkpoint_path = "/path/to/checkpoint"
            load_checkpoint(compiled_model, checkpoint_path) # Your checkpoint loading
            recorder.checkpoint_loaded(path=checkpoint_path)

        # --- Training Loop ---
        num_steps = 1000
        for current_step in range(num_steps):
            # Data Loading for the step
            recorder.data_loading_started(step_number=current_step)
            data_batch = get_next_batch(dataset_iterator) # Your data loading
            recorder.data_loading_completed(step_number=current_step, items_loaded=len(data_batch))

            # Training Step
            recorder.step_started(step_number=current_step)
            loss = training_step(compiled_model, optimizer, data_batch) # Your train step
            recorder.step_completed(step_number=current_step, metrics={"loss": loss}) # Corrected: loss.item() if tensor

            # Optional: Save Checkpoint periodically
            if (current_step + 1) % 100 == 0:
                checkpoint_save_path = f"/path/to/checkpoints/step_{current_step}"
                save_checkpoint(compiled_model, checkpoint_save_path) # Your checkpoint saving
                recorder.checkpoint_saved(path=checkpoint_save_path)
            
            # Example of a custom badput event
            if check_for_system_issues():
                 recorder.custom_badput_event(
                    event_name="SystemPerformanceDegradation",
                    details="Observed high network latency.",
                    severity="WARNING"
                )


        # --- Job Termination ---
        recorder.job_terminated(status='COMPLETED', details='Training finished successfully.')

    except UserCancellationException: # Custom exception for user cancellation
        recorder.user_terminated(timestamp_ms=int(time.time() * 1000))
        # Or recorder.job_terminated(status='USER_CANCELLED', timestamp_ms=int(time.time() * 1000))
    except PreemptionException: # Custom exception for preemption
        recorder.job_terminated(status='PREEMPTED', details='Job was preempted.', timestamp_ms=int(time.time() * 1000))
    except Exception as e:
        recorder.job_terminated(status='FAILED', details=str(e), timestamp_ms=int(time.time() * 1000))
        raise # Re-raise the exception after logging
    finally:
        # Ensure any final cleanup for the recorder if necessary
        # recorder.close() # If applicable
        pass

if __name__ == "__main__":
    # Ensure ml-goodput-measurement is importable in your environment
    # This might involve setting PYTHONPATH or installing the package.
    
    # Note: The following are conceptual placeholders used in main_training_loop
    # and would need actual implementations or imports.
    # from ml_goodput_measurement import GoodputRecorder, Event 
    # import time
    # class UserCancellationException(Exception): pass
    # class PreemptionException(Exception): pass
    # def get_user_schedule_time_from_environment(): return int(time.time() * 1000) - 10000
    # def initialize_tpu_system(): print("TPU Init")
    # def create_model(): print("Create Model"); return "model"
    # def create_optimizer(): print("Create Opt"); return "opt"
    # def compile_model_for_tpu(model): print("Compile"); return model
    # def should_load_checkpoint(): return False
    # def load_checkpoint(model, path): print(f"Load CP from {path}")
    # class DatasetIterator: 
    #     def __init__(self): self.count = 0
    #     def __next__(self): self.count += 1; print(f"Load data {self.count}"); return [self.count] 
    # dataset_iterator = DatasetIterator()
    # def get_next_batch(iterator): return next(iterator)
    # def training_step(model, optimizer, batch): print(f"Train step with {batch}"); return 0.5 
    # def save_checkpoint(model, path): print(f"Save CP to {path}")
    # def check_for_system_issues(): return False

    main_training_loop()
```

## 5. Importability of ml-goodput-measurement

For the `GoodputRecorder` to be used, the `ml-goodput-measurement` library must be installed and importable within the Python environment where your training script executes.

This typically means:
*   **Installing the package:** `pip install ml-goodput-measurement` (if it's a distributable package).
*   **Setting PYTHONPATH:** If you are working with a local version of the library, ensure the directory containing `ml_goodput_measurement` is in your `PYTHONPATH`.
*   **Including it in your Docker image / execution environment:** If running in containers or managed environments, ensure the library is part of the environment setup.

---
