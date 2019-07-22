FROM mkaichen/bazel_cpp:latest

RUN pip3 install tensorflow

ENV APP_DIR /usr/src/tenncor

RUN mkdir -p $APP_DIR
WORKDIR $APP_DIR

COPY . $APP_DIR
RUN pip3 install -r requirements.txt

CMD [ "./tests.sh" ]
