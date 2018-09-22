#!/usr/bin/env bash

# generate and get coverage.dat paths according to arguments
PATHS_STR=$(make $1 2>/dev/null | sed -rn 's/.*COVERAGE_OUTPUT_FILE=(.*)\/coverage\.dat.*/\1/p')
OUTFILE=$2

IFS=$'\n' CPATHS=($PATHS_STR)

# make paths absolute
for ((i=0; i<${#CPATHS[@]}-1; i++));
do
    if [ -d "${CPATHS[i]}" ]; then
        CPATHS[i]=$(realpath "${CPATHS[i]}")
    fi
done

# make all paths unique
IFS=$' ' UCPATHS=($(printf "%s " "${CPATHS[@]}" | sort -u))

# start adding tracefiles together
rm -f "$OUTFILE"
for CPATH in "${UCPATHS[@]}"
do
    if [ -d "$CPATH" ]; then
        CFILE="$CPATH/coverage.dat"
        echo "Processing file $CFILE"
        lcov --remove "$CFILE" 'external/*' '**/test/*' -o "/tmp/$OUTFILE"
        if [ -f "$OUTFILE" ]; then
            lcov -a "$OUTFILE" -a "/tmp/$OUTFILE" -o "/tmp/$OUTFILE"
        fi
        mv "/tmp/$OUTFILE" "$OUTFILE"
        lcov --list $OUTFILE
    fi
done
