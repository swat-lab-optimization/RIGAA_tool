# TODO Create an abstract class to host the logic for calling the validation and the timing
# Use this class to output tests in predefined locations after execution
import logging as log
import random
import time
import sys
import os

from abc import ABC, abstractmethod

from simulator.code_pipeline.validation import TestValidator
from simulator.code_pipeline.tests_generation import TestGenerationStatistic
from simulator.code_pipeline.test_analysis import compute_all_features
from simulator.self_driving.simulation_data import SimulationDataRecord


class Budget:

    # Note: We assume the values are validated at this point
    # TODO: Is this ok?
    def __init__(self, time_budget=None):
        self.time_budget = time_budget
        self.start_real_time = None

    def start(self):
        self.start_real_time = time.perf_counter()

    def get_start_time(self):
        return self.start_real_time

    # We need to ensure that we always return the same type of object:
    def get_remaining_time(self):
        return {"time-budget": self.get_remaining_real_time()}

    def get_remaining_real_time(self):
        log.debug("Remaining real time budget: ", self.time_budget - (time.perf_counter() - self.start_real_time))
        return self.time_budget - (time.perf_counter() - self.start_real_time)

    def can_run_a_test(self):
        return self.get_remaining_real_time() > 0

    # General Check
    def is_over(self):
        return self.get_remaining_real_time() <= 0


class AbstractTestExecutor(ABC):
    start_time = None

    def __init__(self, result_folder, map_size,
                 time_budget=None,
                 road_visualizer=None, debug=False):

        self.debug = debug

        self.result_folder = result_folder

        self.stats = TestGenerationStatistic()
        # Setup the internal logic to check the time budget
        self.time_budget = Budget(time_budget=time_budget)
        # Start counting for passed time
        self.time_budget.start()

        # Variable to capture test generation time
        self.start_generation_time = self.time_budget.get_start_time()

        self.test_validator = TestValidator(map_size)
        self.road_visualizer = road_visualizer

        self.timeout_forced = False

        super().__init__()

    def is_force_timeout(self):
        return self.timeout_forced == True

    def store_test(self, the_test):
        output_file_name = os.path.join(self.result_folder, ".".join(["test", str(the_test.id).zfill(4), "json"]))
        with open(output_file_name, 'w') as test_file:
            test_file.write(the_test.to_json())

    def execute_test(self, the_test):
        # Mark that generation is over and log generation time
        elapsed_generation_time = time.perf_counter() - self.start_generation_time
        # Update the statistics of the run
        self.stats.test_generation_real_times.append(elapsed_generation_time)

        # Reset timer
        self.start_generation_time = None

        # Before executing the test check whether we still have some simulation time (possibly inaccurate) for running
        #   the_test
        if not self.time_budget.can_run_a_test():
            self.timeout_forced = True
            log.warning("Time budget is over, cannot run more tests. FORCE EXIT")
            # Ensure that we do not restart the generation counter
            sys.exit(123)

        # Update the statistics of the run
        self.stats.test_generated += 1

        is_valid, validation_msg = self.validate_test(the_test)

        # This might be placed inside validate_test, but we keep it out since validate_test can be also called by client
        the_test.set_validity(is_valid, validation_msg)

        # Visualize the road if a road visualizer is defined. Also includes results for the validation
        if self.road_visualizer:
            self.road_visualizer.visualize_road_test(the_test)

        if is_valid:
            # Update the statistics of the run
            self.stats.test_valid += 1

            start_execution_real_time = time.perf_counter()
            try:
                test_outcome, description, execution_data = self._execute(the_test)
            finally:
                # Log time also on error
                real_time_elapsed = time.perf_counter() - start_execution_real_time
                # Update the statistics of the run
                self.stats.test_execution_real_times.append(real_time_elapsed)

            # Check that at least one element is there
            if execution_data and len(execution_data) > 0:
                simulated_time_elapsed = execution_data[-1].timer

                if self.time_budget.is_over():
                    # The last test should not be considered since it went OVER budget
                    log.info("Run overbudget discard the last experiment. FORCE EXIT")
                    self.timeout_forced = True
                    # Ensure that we do not restart the generation counter
                    sys.exit(123)
                else:
                    # Update the statistics of the run
                    self.stats.test_execution_simulation_times.append(simulated_time_elapsed)

            # Decorating the_test with additional attributes
            setattr(the_test, 'execution_data', execution_data)
            setattr(the_test, 'test_outcome', test_outcome)
            setattr(the_test, 'description', description)

            # Compute the features dict of this test and
            features = compute_all_features(the_test, execution_data)

            log.info("Features: {}".format(features))

            # Decorating the_test with the features
            setattr(the_test, 'features', features)

            # Store the generated tests into the result_folder as JSON file
            self.store_test(the_test)

            # TODO Move to decorator?
            if test_outcome == "ERROR":
                # Update the statistics of the run
                self.stats.test_in_error += 1
                # This indicates a generic error during the execution, usually caused by a malformed test that the
                # validation logic was not able to catch.
                # Ensure we define this object
                execution_data = []
            elif test_outcome == "PASS":
                # Update the statistics of the run
                self.stats.test_passed += 1
            else:
                # Update the statistics of the run
                self.stats.test_failed += 1
                # Valid, either pass or fail
                if description.startswith("Car drove out of the lane "):
                    self.stats.obes += 1
        else:
            # Store the generated tests into the result_folder even if it is not valid to keep track of them
            # and also let clients debug. Note: we do not compute structural features for invalid tests
            self.store_test(the_test)
            # Update the statistics of the run
            self.stats.test_invalid += 1
            test_outcome = "INVALID"
            description = validation_msg
            execution_data = []

        # Mark that generation is restarted.
        self.start_generation_time = time.perf_counter()

        return test_outcome, description, execution_data

    def validate_test(self, the_test):
        log.debug("Validating test")
        return self.test_validator.validate_test(the_test)

    def get_stats(self):
        return self.stats

    def close(self):
        self._close()

    def is_over(self):
        return self.time_budget.is_over()

    # This is mostly for debugging purposes
    def get_remaining_time(self):
        return self.time_budget.get_remaining_time()

    @abstractmethod
    def _close(self):
        if not self.time_budget.is_over():
            log.warning("Despite the time budget is not over executor is exiting!")

    @abstractmethod
    def _execute(self, the_test):
        # This should not be necessary, but better safe than sorry... The point here is that one can have spent a lot
        # of time validing the tests?
        if not self.time_budget.can_run_a_test():
            self.timeout_forced = True
            log.warning("Time budget is over, cannot run more tests. FORCE EXIT")
            sys.exit(123)


class MockExecutor(AbstractTestExecutor):

    def _execute(self, the_test):
        # Ensure we do not execute anything longer than the time budget
        super()._execute(the_test)

        test_outcome = random.choice(["FAIL", "FAIL", "FAIL", "PASS", "PASS", "PASS", "PASS", "PASS", "ERROR"])
        description = "Mocked test results"

        sim_state = SimulationDataRecord(
            timer=3.0,
            pos=[0.0, 0.0, 1.0],
            dir=[0.0, 0.0, 1.0],
            vel=[0.0, 0.0, 1.0],
            steering=0.0,
            steering_input=0.0,
            brake=0.0,
            brake_input=0.0,
            throttle=0.0,
            throttle_input=0.0,
            wheelspeed=0.0,
            vel_kmh=0.0,
            is_oob=False,
            oob_counter=0,
            max_oob_percentage=0.0,
            oob_distance=0.0,
            oob_percentage=50.0
        )

        execution_data = [sim_state]

        log.info("Pretend test is executing for 5 seconds.")
        time.sleep(5)

        return test_outcome, description, execution_data

    def _close(self):
        super()._close()
        print("Closing Mock Executor")
