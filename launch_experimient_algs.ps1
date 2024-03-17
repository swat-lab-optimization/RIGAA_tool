# Define the arguments
$args = @(
    "--problem", "vehicle",
    "--algorithm", "rigaa",
    "--runs", "3",
    "--save_results", "True",
    "--eval_time", "02:05:00",
    "--full", "True"
)

# Define the different algorithm options
$algorithms = @( "rigaa", "nsga2", "smsemoa", "rigaa_s", "random")

# Loop through different values of "--algorithm"
foreach ($algorithm in $algorithms) {
    # Update the value of "--algorithm" in the arguments
    $args[$args.IndexOf("--algorithm") + 1] = $algorithm
    # Execute the optimize.py script with updated arguments
    python optimize.py $args
}
#powershell -ExecutionPolicy Bypass -File optimize.ps1
