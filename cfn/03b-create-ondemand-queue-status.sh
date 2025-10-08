#!/bin/bash

REGION=$1

if [[ "$REGION" == "" ]];
then
	echo "ERROR: must provide AWS region code as argument to script"
	exit
fi

aws cloudformation describe-stacks --region ${REGION} --stack-name vf-ondemand --query "Stacks[0].StackStatus"
