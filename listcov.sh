#!/usr/bin/env bash

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
set +x
for CPATH in "${UCPATHS[@]}"
do
    if [ -d "$CPATH" ]; then
        CFILE="$CPATH/coverage.dat"
        echo "+processing file $CFILE"
        lcov --remove "$CFILE" 'external/*' -o "/tmp/$OUTFILE"
        if [ -f "$OUTFILE" ]; then
            lcov "$OUTFILE" "/tmp/$OUTFILE" -o "/tmp/$OUTFILE"
        fi
        mv "/tmp/$OUTFILE" "$OUTFILE"
    fi
done
