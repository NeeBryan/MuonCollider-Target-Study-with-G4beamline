#!/bin/bash

####################################################################################################
# Description:
# This script automates the setup, simulation, and plotting process for a parametric study of 
# a graphite target geometry using G4beamline. It loops over a set of input angular parameters 
# (theta, alpha, beta) to:
#
# 1. Create or reuse geometry directories named based on those angles.
# 2. Copy a template geometry if needed.
# 3. Modify the `GraphiteTarget.g4bl` input file with the corresponding theta, alpha, and beta values.
# 4. Run the G4beamline simulation for each configuration.
# 5. Update and run the associated plotting script (`GraphiteTarget_Plot.py`) to generate plots.
# 6. Move the resulting plots into an `Image/` subdirectory for each configuration.
#
# Notes:
# - Uses a "sanitize" function to convert float values into filename-safe strings (e.g., 0.5 → 0pt5).
# - Assumes the existence of a template directory: Target_Default_Directory/Target_Geometry_Cylinder.
# - Requires G4beamline and Python to be installed and accessible in the environment.
#
# Author: [Cheng-Hsu (Bryan) Nee]
# Date: [2025-07-21]
####################################################################################################

# Base directory
BASE_DIR=$(pwd)

# Parameter lists
theta_list=(10)
alpha_list=(10)
beta_list=(0 0.5 1 2)

# Template directory to copy from if needed
TEMPLATE_DIR="Target_Default_Directory/Target_Geometry_Cylinder"

# Function to convert float to dir-safe string (e.g., 0.5 -> 0pt5)
sanitize() {
    echo "$1" | sed 's/-/neg/' | sed 's/\./pt/'
}

# Check theta and alpha lists are the same length
if [ "${#theta_list[@]}" -ne "${#alpha_list[@]}" ]; then
    echo "Error: theta_list and alpha_list must be the same length."
    exit 1
fi

# Loop over zipped theta-alpha and beta
for ((i=0; i<${#theta_list[@]}; i++)); do
    theta="${theta_list[$i]}"
    alpha="${alpha_list[$i]}"
    theta_str=$(sanitize "$theta")
    alpha_str=$(sanitize "$alpha")

    for beta in "${beta_list[@]}"; do
        beta_str=$(sanitize "$beta")

        # Construct directory name
        dir="Target_Rotate_theta_${theta_str}_Beam_alpha_${alpha_str}_beta_${beta_str}"
        geometry_dir="$dir/Target_Geometry_Cylinder"

        echo "Processing: $dir (theta=$theta, alpha=$alpha, beta=$beta)"

        # If directory doesn't exist, copy from template
        if [[ ! -d "$geometry_dir" ]]; then
            echo "  -> Directory doesn't exist. Creating from $TEMPLATE_DIR"
            mkdir -p "$dir"
            cp -r "$TEMPLATE_DIR" "$geometry_dir"
        else
            echo "  -> Directory already exists. Skipping copy."
        fi

        # Change into the geometry directory
        cd "$geometry_dir" || continue

        # Modify parameters in Graphite.g4bl (lines 11–13)
        sed -i "s/param theta=0/param theta=$theta/" GraphiteTarget.g4bl
        sed -i "s/param alpha=0/param alpha=$alpha/" GraphiteTarget.g4bl
        sed -i "s/param beta=0/param beta=$beta/" GraphiteTarget.g4bl

        # Run g4bl
        rm solenoid_*
        rm FieldAlongZ.txt
        g4bl GraphiteTarget.g4bl

        # Update CSV paths in Graphite_Plot.py
	    sed -i "s|\"./Target_Rotate_theta_0_Beam_alpha_0_beta_0_\(.*\.csv\)\"|\"./Target_Rotate_theta_${theta_str}_Beam_alpha_${alpha_str}_beta_${beta_str}_\1\"|g" GraphiteTarget_Plot.py

	    # Update LaTeX label
        sed -i "s|r = 30mm, \\\$\\\\theta = 0, \\\\alpha = 0, \\\\beta = 0\\\$|r = 30mm, \\\$\\\\theta = $theta, \\\\alpha = $alpha, \\\\beta = $beta\\\$|g" GraphiteTarget_Plot.py
        
        # Run the Python plotting script
        python3 GraphiteTarget_Plot.py

        # Move plots to Image directory
        mkdir -p "$BASE_DIR/$geometry_dir/Image"
        mv *.png "$BASE_DIR/$geometry_dir/Image/"

        # Return to base directory
        cd "$BASE_DIR"
    done
done

