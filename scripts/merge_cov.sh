#!/usr/bin/env bash

MYNAME="${0##*/}";

function usage {
    cat <<EOF
synopsis: filter for trace files found by COVERAGE_OUTPUT_FILE=<file/path>
format in stdin, then stitch tracefiles together as outfile.

usage: stdin | $MYNAME outfile
EOF;
    exit 1;
}

PIPE_IN="";
if [ -p /dev/stdin ];
then
    PIPE_IN=$(</dev/stdin);
else
    echo "Missing STDIN pipe";
    usage;
fi

if [ "$#" -lt 1 ];
then
    echo "Missing outfile argument";
    usage;
fi

OUTFILE=$1;

# extract coverage paths
PATHS_STR="$(echo "$PIPE_IN" | sed -rn 's/.*COVERAGE_OUTPUT_FILE=(.*)\/coverage\.dat.*/\1/p')";

IFS=$'\n';
CPATHS=($PATHS_STR);

# make paths absolute
for ((i=0; i<${#CPATHS[@]}; i++))
do
    if [ -d "${CPATHS[i]}" ];
    then
        CPATHS[i]=$(realpath "${CPATHS[i]}");
    fi
done

# make all paths unique
IFS=$' ';
UCPATHS=($(printf "%s " "${CPATHS[@]}" | sort -u));

# start stitching tracefiles together
rm -f "$OUTFILE";
for CPATH in "${UCPATHS[@]}"
do
    if [ -d "$CPATH" ];
    then
        CFILE="$CPATH/coverage.dat";
        echo "Stitching file $CFILE";
        if [ -f "$OUTFILE" ];
        then
            lcov -a "$CFILE" -a "$OUTFILE" -o "$OUTFILE";
        else
            cp "$CFILE" "$OUTFILE";
        fi
    fi
done
