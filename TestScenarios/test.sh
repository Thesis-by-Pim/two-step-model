#!/usr/bin/env bash

script_dir=$(dirname "$0")
# Build array of scenario .ini files
mapfile -t scenarios < <(find ${script_dir} -type f -name '*.ini' | sort)

for scenario in "${scenarios[@]}"; do
    outfolder="$(dirname ${scenario})/output"
    logfolder="${outfolder}/logs"
    formatted_date=$(date +"%y-%m-%d %H:%M:%S")

    # Check if the log folder exists
    if [ ! -d "$logfolder" ]; then
        # Directory does not exist, create it
        mkdir -p "$logfolder"
        echo "Directory created: $logfolder"
    fi

    echo "Running scenario: ${scenario}"
    start_time=$(date +%s.%N)

    # Run in subshell in background:
    (
        cd ${script_dir}
        cd ..
        
        pipenv run cli --verbosity DEBUG run --solver NUTRITION_AND_MARKET_VALUE --plot 0 --config ${scenario} > "${logfolder}/${formatted_date}.log"
        # pipenv run cli run --solver NUTRITION_AND_MARKET_VALUE --plot 0 --config ${scenario} &> "${logfolder}/${formatted_date}.log"
    ) &

    while ps -p $! >/dev/null; do
        current_time=$(date +%s.%N)
        elapsed_time=$(echo "$current_time - $start_time" | bc)
        
        echo -ne "Elapsed time: $elapsed_time seconds\r"
        sleep 0.1
    done
    echo -ne "Complete after: $elapsed_time seconds\r"
    echo ""
    echo ""
    echo ""

done