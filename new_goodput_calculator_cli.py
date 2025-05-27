# new_goodput_calculator_cli.py
"""
Script to calculate and display job goodput using GoodputCalculator.

This script demonstrates how to use the GoodputCalculator from the
ml-goodput-measurement library to fetch and display goodput metrics
for a given training job.

Example Usage:
    python new_goodput_calculator_cli.py --job_name "my_training_job_123"
    python new_goodput_calculator_cli.py --job_name "my_training_job_456" --logger_name "custom_logger_for_job_456"

The script assumes that the necessary event data for the specified job_name
(or logger_name) is available in the backend data store that GoodputCalculator
is configured to use (e.g., BigQuery).
"""

import argparse
import datetime
import json # For JSON export placeholder

# Assuming GoodputCalculator and GoodputEventName are importable
# from your ml_goodput_measurement library structure.
# Adjust the import paths if your library structure is different.
try:
    from ml_goodput_measurement.goodput import GoodputCalculator
    from ml_goodput_measurement.goodput_utils import GoodputEventName # For pretty printing breakdown
except ImportError:
    # Fallback for cases where the exact structure might vary or for testing
    # This allows the script to be created even if the lib isn't in PYTHONPATH yet
    # In a real scenario, this try/except for imports might not be ideal.
    print("Warning: ml_goodput_measurement modules not found. Using placeholder classes.")
    class GoodputCalculator:
        def __init__(self, logger_name):
            self.logger_name = logger_name
            print(f"Placeholder GoodputCalculator initialized for {logger_name}")
        def get_job_goodput(self, include_badput_breakdown=False):
            print(f"Placeholder get_job_goodput called for {self.logger_name}")
            # Return a mock GoodputResult-like object
            class MockGoodputResult:
                pass
            res = MockGoodputResult()
            res.goodput_percent = 75.0
            res.badput_breakdown_percent = {GoodputEventName.DATA_LOADING: 50.0, GoodputEventName.TPU_INIT: 50.0} if include_badput_breakdown else {}
            res.job_start_time_ms = (datetime.datetime.now() - datetime.timedelta(hours=1)).timestamp() * 1000
            res.job_end_time_ms = datetime.datetime.now().timestamp() * 1000
            res.last_step_number = 1000
            res.last_step_time_ms = (datetime.datetime.now() - datetime.timedelta(minutes=5)).timestamp() * 1000
            res.total_goodput_time_ms = 2700 * 1000
            res.total_badput_time_ms = 900 * 1000
            res.total_measured_duration_ms = 3600 * 1000
            return res

    class GoodputEventName: # Placeholder Enum
        DATA_LOADING = "DATA_LOADING"
        TPU_INIT = "TPU_INIT"
        TRAINING_PREP = "TRAINING_PREP"
        # Add other event names as needed for the placeholder


def main():
    parser = argparse.ArgumentParser(description="Calculate and display job goodput.")
    parser.add_argument(
        "--job_name",
        type=str,
        required=True,
        help="The name of the job to calculate goodput for."
    )
    parser.add_argument(
        "--logger_name",
        type=str,
        help="Optional. The specific logger name to use. If not provided, "
             "it will be derived from job_name (e.g., 'goodput_<job_name>')."
    )
    # Add other potential arguments here, for example, to specify
    # a date range for event fetching if your GoodputCalculator supports it.
    # parser.add_argument("--start_date", type=str, help="Start date for events (YYYY-MM-DD)")
    # parser.add_argument("--end_date", type=str, help="End date for events (YYYY-MM-DD)")

    args = parser.parse_args()

    job_name = args.job_name
    # Derive logger_name if not provided, following common practice
    logger_name = args.logger_name if args.logger_name else f'goodput_{job_name}'

    print(f"Attempting to calculate goodput for job: '{job_name}' using logger: '{logger_name}'\n")

    try:
        # 1. Instantiate GoodputCalculator
        # The constructor might take other parameters depending on your
        # library's configuration (e.g., project_id, dataset_id for BigQuery).
        # For this template, we assume a simple instantiation.
        calculator = GoodputCalculator(logger_name=logger_name)
        # This print statement is more for confirming which calculator (real or placeholder) is used.
        # print(f"GoodputCalculator instantiated for logger: '{calculator.logger_name}'")


        # 2. Call get_job_goodput
        # include_badput_breakdown=True will provide detailed badput sources.
        goodput_result = calculator.get_job_goodput(include_badput_breakdown=True)

        # 3. Print the results in a user-friendly format
        if goodput_result:
            print("\n--- Goodput Calculation Results ---")
            print(f"  Goodput Percentage: {goodput_result.goodput_percent:.2f}%")

            if goodput_result.badput_breakdown_percent:
                print("\n  Badput Breakdown (by percentage of total badput time):")
                for event_enum, percentage in goodput_result.badput_breakdown_percent.items():
                    # Use .name attribute if event_enum is an Enum, otherwise convert to string
                    event_name_str = event_enum.name if hasattr(event_enum, 'name') and not isinstance(event_enum, str) else str(event_enum)
                    print(f"    - {event_name_str}: {percentage:.2f}%")
            else:
                print("\n  No badput breakdown available or badput is zero.")

            print("\n  Job Timeline Information:")
            print(f"    Job Start Time: {datetime.datetime.fromtimestamp(goodput_result.job_start_time_ms / 1000) if goodput_result.job_start_time_ms else 'N/A'}")
            print(f"    Job End Time: {datetime.datetime.fromtimestamp(goodput_result.job_end_time_ms / 1000) if goodput_result.job_end_time_ms else 'N/A'}")
            print(f"    Last Step Number: {goodput_result.last_step_number if goodput_result.last_step_number is not None else 'N/A'}")
            print(f"    Last Step Time: {datetime.datetime.fromtimestamp(goodput_result.last_step_time_ms / 1000) if goodput_result.last_step_time_ms else 'N/A'}")
            
            print(f"\n  Total Goodput Time (seconds): {goodput_result.total_goodput_time_ms / 1000:.2f}")
            print(f"  Total Badput Time (seconds): {goodput_result.total_badput_time_ms / 1000:.2f}")
            print(f"  Total Measured Duration (seconds): {goodput_result.total_measured_duration_ms / 1000:.2f}")

            # 4. Placeholder for JSON export
            # To enable, uncomment the block below and ensure goodput_result attributes match.
            # The example below assumes badput_breakdown_percent keys might be enums.
            """
            goodput_result_dict = {
                "job_name": job_name,
                "logger_name": logger_name,
                "goodput_percent": goodput_result.goodput_percent,
                "badput_breakdown_percent": { 
                    (k.name if hasattr(k, 'name') and not isinstance(k, str) else str(k)): v 
                    for k, v in goodput_result.badput_breakdown_percent.items()
                },
                "job_start_time_ms": goodput_result.job_start_time_ms,
                "job_end_time_ms": goodput_result.job_end_time_ms,
                "last_step_number": goodput_result.last_step_number,
                "last_step_time_ms": goodput_result.last_step_time_ms,
                "total_goodput_time_ms": goodput_result.total_goodput_time_ms,
                "total_badput_time_ms": goodput_result.total_badput_time_ms,
                "total_measured_duration_ms": goodput_result.total_measured_duration_ms,
                # If your goodput_result object has raw_events and they are serializable:
                # "raw_events": [event._asdict() for event in goodput_result.raw_events] 
            }
            output_filename = f"{job_name}_goodput_results.json"
            try:
                with open(output_filename, 'w') as f:
                    json.dump(goodput_result_dict, f, indent=4)
                print(f"\nResults also saved to {output_filename}")
            except Exception as e:
                print(f"Error saving results to JSON: {e}")
            """
            print("\n(JSON export placeholder - uncomment and modify code in script to enable)")

        else:
            print("Could not retrieve goodput information. This might be due to:")
            print("  - No events found for the given logger_name.")
            print("  - The job not having a JOB_STARTED or JOB_TERMINATED event.")
            print("  - Issues connecting to the backend data store used by GoodputCalculator.")

    except ImportError as e:
        # This specific error regarding ml_goodput_measurement is handled by the placeholder at the top for now.
        # If the placeholder wasn't there, this would be the primary way to catch it.
        print(f"Error: Could not import actual ml_goodput_measurement modules. {e}")
        print("Please ensure the library is installed and accessible in your PYTHONPATH.")
    except Exception as e:
        print(f"An error occurred: {e}")
        # For more detailed debugging, uncomment the following lines:
        # import traceback
        # traceback.print_exc()

if __name__ == "__main__":
    main()
```
