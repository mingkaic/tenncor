FROM mkaichen/bazel_cpp:latest

ENV APP_DIR /usr/src/tenncor

RUN mkdir -p $APP_DIR
WORKDIR $APP_DIR

COPY . $APP_DIR
RUN pip install -r requirements.txt
RUN bazel build --config gcc_eigen_optimal //:tenncor_py
