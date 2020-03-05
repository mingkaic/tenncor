FROM mkaichen/tenncor-build:latest

FROM ubuntu:18.04

RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y python3.6 python3-pip python3.6-dev g++

COPY --from=0 /usr/src/tenncor/bazel-bin/tenncor.so /usr/lib/python3/dist-packages
