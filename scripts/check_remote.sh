#!/usr/bin/env bash

function check_status() {
    STATUS=$(curl --write-out "%{http_code}\n" --silent --output /dev/null $1);
    if [ "$STATUS" -ne $2 ];
    then
        curl $1;
        exit 1;
    fi
}
