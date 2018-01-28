FROM mkaichen/bazel_cpp:latest

ENV APP_DIR /usr/src/tenncor
ENV GTEST_REPEAT 5
ENV GTEST_BREAK_ON_FAILURE 1 
ENV GTEST_SHUFFLE 1

RUN mkdir -p $APP_DIR
WORKDIR $APP_DIR

COPY . $APP_DIR

CMD [ "bash", "tests.sh" ]
