FROM mkaichen/bazel_cpp:latest

ENV APP_DIR /usr/src/tenncor

RUN mkdir -p $APP_DIR
WORKDIR $APP_DIR

COPY . $APP_DIR

ENV GTEST_REPEAT 5
ENV export GTEST_BREAK_ON_FAILURE 1 
ENV GTEST_SHUFFLE 1

CMD [ "bash", "tests.sh" ]
