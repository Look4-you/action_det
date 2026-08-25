#!/bin/bash
sleep_second=1

if [ "$GPU_OLD" == "1" ]; then
    #旧架构的GPU，例如T4, V100，不支持flash-attn
    pip uninstall flash-attn -y
fi

if [ -n "$SLEEP_SECOND" ]; then
    sleep_second=$SLEEP_SECOND
fi

sleep $sleep_second

http_server_threads=10

if [ -n "$HTTP_SERVER_THREADS" ]; then
    http_server_threads=$HTTP_SERVER_THREADS
fi

gunicorn -w 1 --threads $http_server_threads -b 0.0.0.0:9000 service_http:app
