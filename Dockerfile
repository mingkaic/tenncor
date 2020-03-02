FROM mkaichen/bazel_cpp:latest

ENV APP_DIR /usr/src/tenncor

RUN mkdir -p $APP_DIR
WORKDIR $APP_DIR

COPY . $APP_DIR
RUN apt-get update && apt-get install -y curl
RUN pip3 install numpy pyyaml

RUN bazel build --config gcc_eigen_optimal //:tenncor_py
