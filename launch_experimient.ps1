# Define the arguments
$args = @(
    "--problem", "vehicle",
    "--algorithm", "rigaa",
    "--ro", "0.0",
    "--runs", "5",
    "--save_results", "True",
    "--eval_time", "02:05:00",
    "--full", "True"
)
# Loop through different values of "--ro" from 0 to 1 with a step of 0.2
for ($ro = 1; $ro -le 1; $ro += 0.2) {
    # Update the value of "--ro" in the arguments
    $args[$args.IndexOf("--ro") + 1] = $ro
    # Execute the optimize.py script with updated arguments
    python optimize.py $args
}