# Define the arguments
$args = @(
    "--problem", "vehicle",
    "--algorithm", "rigaa",
    "--runs", "5",
    "--save_results", "True",
    "--eval_time", "02:05:00",
    "--full", "True",
    "--ro", "0.2",
)

# Define the different algorithm options
$algorithms = @( "rigaa",  "rigaa_s", "smsemoa") #"nsga2", "smsemoa",

# Loop through different values of "--algorithm"
foreach ($algorithm in $algorithms) {
    # Update the value of "--algorithm" in the arguments
    $args[$args.IndexOf("--algorithm") + 1] = $algorithm
    # Execute the optimize.py script with updated arguments
    python optimize.py $args
}
#powershell -ExecutionPolicy Bypass -File optimize.ps1
