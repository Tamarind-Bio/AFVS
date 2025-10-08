#!/bin/bash

REGION=$1

if [[ "$REGION" == "" ]];
then
	echo "ERROR: must provide AWS region code as argument to script"
	exit
fi


if [[ ! -e yaml/vf-ondemand-queue.yaml ]];
then
	echo "ERROR: yaml/vf-ondemand-queue.yaml is not setup yet"
	exit;
fi

if [[ ! -e params/${REGION}/vf-ondemand-parameters.json ]];
then
	echo "ERROR: params/${REGION}/vf-ondemand-parameters.json is not setup yet"
	exit;
fi


aws cloudformation create-stack --stack-name vf-ondemand \
--template-body file://yaml/vf-ondemand-queue.yaml \
--parameters file://params/${REGION}/vf-ondemand-parameters.json \
--region ${REGION}
