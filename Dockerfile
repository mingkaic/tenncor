FROM mkaichen/bazel_cpp:latest

ENV APP_DIR /usr/src/tenncor

RUN mkdir -p $APP_DIR
WORKDIR $APP_DIR

EXTEND . $APP_DIR

CMD [ "bash", "tests.sh" ]
