#!/bin/bash

####################################################################################################
# Description:
# This script automates the geometry generation and simulation process for a parametric study of a 
# rotated graphite target in G4beamline. It iterates over combinations of rotation angles:
#
#   - theta (target rotation angle)
#   - alpha (beam incidence angle)
#   - beta (target twist angle)
#
# For each unique (theta, alpha, beta) configuration:
# 1. It constructs a directory named using the sanitized values of the angles.
# 2. If the directory does not already exist, it is created by copying from a predefined template.
# 3. The `GraphiteTarget.g4bl` file is updated in-place with the new parameter values.
# 4. The G4beamline simulation is run for that configuration.
#
# Notes:
# - Uses a sanitize function to convert float values to safe strings for directory names.
# - Assumes the existence of a template directory: Target_Default_Directory/Target_Geometry_Cylinder.
# - Requires G4beamline to be installed and accessible from the environment.
# - No plotting is performed in this version of the script (unlike the previous one).
#
# Author: [Your Name]
# Date: [YYYY-MM-DD]
####################################################################################################


# Base directory
BASE_DIR=$(pwd)

# Parameter lists
theta_list=(3 5 8 10)
alpha_list=(3 5 8 10)
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

        # Return to base directory
        cd "$BASE_DIR"
    done
done

