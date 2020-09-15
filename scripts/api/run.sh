#!/bin/bash

# Check virtual environment exist
if [ ! -d "venv" ]
then
    printf "[ERROR]: Virtual environment is not ready. Please create virtual environment by run \'bash scripts/venv.sh\'\n"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate
printf "[INFO]: Activated virutal environment\n"

# Get IP and PORT
IP=$1
PORT=$2

# Check IP or PORT not assigned
if [ -z "$IP" ] || [ -z $PORT ]
then 
    printf "[ERROR]: IP or PORT is not assigned. Example: bash scripts/api/run.sh 127.0.0.1 8000\n"
    exit 1
fi

printf "[INFO]: IP:$IP \tPORT:$PORT\n"

# Run API server
printf "[INFO]: API server is running\n"
python apis/backend/manage.py runserver $IP:$PORT