FROM mkaichen/bazel_cpp:081e0e24563758804fbb7d8b421db12a4674cb60

ENV APP_DIR /usr/src/tenncor
ARG REMOTE_CACHE=""

RUN mkdir -p $APP_DIR
WORKDIR $APP_DIR

COPY . $APP_DIR
RUN pip install -r requirements.txt
RUN bazel build --config gcc_eigen_optimal --remote_http_cache="$REMOTE_CACHE" //:tenncor_py
