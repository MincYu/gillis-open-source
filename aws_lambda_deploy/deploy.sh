#!/bin/bash

JOB="wresnet50_2"

usage() { echo "Usage: cmd [-j job_name]" 1>&2; exit 0; }

while getopts "j:h" opt; do
    case ${opt} in
      j)
        JOB=${OPTARG}
        echo ${JOB}
        ;;
      h | *)
        usage
        ;;
    esac
done

python deploy.py --job ${JOB}
cd ${JOB}/output
sam build
sam deploy --guided
echo ""
echo "Finished"