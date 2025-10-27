#!/bin/bash
# -*- coding: utf-8 -*-
set -euo pipefail

# Loop over each discretization (detected from ijkl-sxytxz-*.asc)
for ijkl_file in ijkl-sztx-*.asc; do
    # Extract the discretization pattern (e.g., "1-2-2")
    disc=$(echo "$ijkl_file" | sed -E 's/ijkl-sztx-([0-9-]+)\.asc/\1/')
    
    # Match the TW1 file with same discretization and Re number
    for twfile in TW2-2pi1piRe*-"$disc"-sol.asc; do
        # Skip if no matching TW2 file
        [[ -e "$twfile" ]] || continue
        
        # Extract base name components
        base=${twfile%-sol.asc}
        sigma_file="${base}-sol-sigma.asc"
        ncfile="u${base}.nc"
        
        # Run projectfield
        echo "Projecting field for $disc -> $ncfile"
        projectfield -x "$twfile" "$ijkl_file" u.nc "${base##*/}"
        # mv u.nc "$ncfile"
        
        # Get Re from the filename (e.g., Re200)
        Re=$(echo "$twfile" | sed -E 's/.*Re([0-9]+).*/\1/')
        
        # Run findsoln
        echo "Finding solution for $base"
		mkdir -p $base
        findsoln -eqb -zrel -T 10 -R "$Re" -sigma "$sigma_file" -symms sztx.asc -od "$base" "$ncfile"
        
        echo "✅ Finished $base"
        echo
    done
done

