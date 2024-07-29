import os
import logging as log
import argparse
import json
def setup_logging(log_to, debug):
    """
    It sets up the logging system
    """

    term_handler = log.StreamHandler()
    log_handlers = [term_handler]
    start_msg = "Started test generation"

    if log_to is not None:
        file_handler = log.FileHandler(log_to, "w", "utf-8")
        log_handlers.append(file_handler)
        start_msg += " ".join([", writing logs to file: ", str(log_to)])

    log_level = log.DEBUG if debug else log.INFO

    log.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=log_level,
        handlers=log_handlers,
        force=True,
    )
    log.info(start_msg)

def parse_arguments():
    """
    This function parses the arguments passed to the script
    :return: The arguments that are being passed to the program
    """

    log.info("Parsing the arguments")
    parser = argparse.ArgumentParser(
        prog="compare.py",
        description="A tool for generating test cases for autonomous systems",
        epilog="For more information, please visit ",
    )
    parser.add_argument(
        "--result_folder",
        #nargs="+",
        type=str,
        help="The source folders of the metadate to analyze",
        required=True,
    )
    parser.add_argument(
        "--save_name",
       # nargs="+",
        type=str,
        help="The names of the corresponding algorithms",
        required=True,
    )

    in_arguments = parser.parse_args()
    return in_arguments


def merge_runs(result_folder, result_save_path):
    """
    This function merges the runs of the algorithms
    :param stats_path: The paths to the metadata
    :param stats_names: The names of the algorithms
    :return: None
    """

    run = 0
    final_results = {}
    for subf in os.listdir(result_folder):
        if "stats" in subf:
            for file in os.listdir(os.path.join(result_folder, subf)):
                if "all_tests" in file:

                    with open(os.path.join(result_folder,subf, file)) as f:
                        results = json.load(f)
                    for res in results:
                        final_results["run"+str(run)] = results[res]
                        run += 1

    with open(os.path.join(result_folder,result_save_path), "w") as f:
        json.dump(final_results, f, indent=4)


if __name__ == "__main__":
    args = parse_arguments()
    setup_logging(None, False)
    merge_runs(args.result_folder, args.save_name)

                

                

