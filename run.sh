#!/bin/bash

APP_PID=
stopRunningProcess() {
    # Based on https://linuxconfig.org/how-to-propagate-a-signal-to-child-processes-from-a-bash-script
    if test ! "${APP_PID}" = '' && ps -p ${APP_PID} > /dev/null ; then
       > /proc/1/fd/1 echo "Stopping Streamlit application with process ID ${APP_PID}"

       kill -TERM ${APP_PID}
       > /proc/1/fd/1 echo "Waiting for Streamlit application to process SIGTERM signal"

        wait ${APP_PID}
        > /proc/1/fd/1 echo "Streamlit application has stopped"
    else
        > /proc/1/fd/1 echo "Streamlit application was not started when the signal was sent or it has already been stopped"
    fi
}

trap stopRunningProcess EXIT TERM

source ${VIRTUAL_ENV}/bin/activate

streamlit run /emoji\ creator\ project/app.py &
APP_PID=${!}

wait ${APP_PID}
