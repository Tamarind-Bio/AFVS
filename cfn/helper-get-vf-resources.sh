#!/bin/bash

# Helper script to retrieve resource IDs/ARNs from the VF stack
# for use in the on-demand queue parameter file

REGION=$1
STACK_NAME=${2:-vf}

if [[ "$REGION" == "" ]];
then
	echo "ERROR: must provide AWS region code as argument to script"
	echo "Usage: $0 <region> [stack-name]"
	exit 1
fi

echo "Retrieving resources from stack '${STACK_NAME}' in region '${REGION}'..."
echo ""

# Get ECS Instance Profile name
INSTANCE_PROFILE=$(aws cloudformation list-stack-resources \
  --region ${REGION} \
  --stack-name ${STACK_NAME} \
  --query "StackResourceSummaries[?LogicalResourceId=='ECSTaskInstanceProfile'].PhysicalResourceId" \
  --output text)

# Get full ARN for instance profile
if [[ "$INSTANCE_PROFILE" != "" ]]; then
  INSTANCE_PROFILE_ARN=$(aws iam get-instance-profile \
    --instance-profile-name ${INSTANCE_PROFILE} \
    --query "InstanceProfile.Arn" \
    --output text 2>/dev/null)
else
  INSTANCE_PROFILE_ARN="ERROR: Could not find ECSTaskInstanceProfile"
fi

# Get Batch Service Role ARN
BATCH_ROLE=$(aws cloudformation list-stack-resources \
  --region ${REGION} \
  --stack-name ${STACK_NAME} \
  --query "StackResourceSummaries[?LogicalResourceId=='BatchInstanceRole'].PhysicalResourceId" \
  --output text)

BATCH_ROLE_ARN=$(aws iam get-role \
  --role-name ${BATCH_ROLE} \
  --query "Role.Arn" \
  --output text 2>/dev/null)

# Get Launch Template ID
LAUNCH_TEMPLATE=$(aws cloudformation list-stack-resources \
  --region ${REGION} \
  --stack-name ${STACK_NAME} \
  --query "StackResourceSummaries[?LogicalResourceId=='BatchLaunchTemplate'].PhysicalResourceId" \
  --output text)

# Get Security Group ID
SECURITY_GROUP=$(aws cloudformation list-stack-resources \
  --region ${REGION} \
  --stack-name ${STACK_NAME} \
  --query "StackResourceSummaries[?LogicalResourceId=='BatchSecurityGroup'].PhysicalResourceId" \
  --output text)

echo "ECS Instance Profile ARN:"
echo "  ${INSTANCE_PROFILE_ARN}"
echo ""
echo "Batch Service Role ARN:"
echo "  ${BATCH_ROLE_ARN}"
echo ""
echo "Launch Template ID:"
echo "  ${LAUNCH_TEMPLATE}"
echo ""
echo "Security Group ID:"
echo "  ${SECURITY_GROUP}"
echo ""
echo "---"
echo ""
echo "Update your params/${REGION}/vf-ondemand-parameters.json file with these values:"
echo ""
cat << EOF
[
  {
    "ParameterKey": "ProjectName",
    "ParameterValue": "AdaptiveFlow-${REGION}"
  },
  {
    "ParameterKey": "VFStackName",
    "ParameterValue": "${STACK_NAME}"
  },
  {
    "ParameterKey": "VPCStackParameter",
    "ParameterValue": "vf-vpc"
  },
  {
    "ParameterKey": "OnDemandInstanceTypes",
    "ParameterValue": "c5.24xlarge,c5.12xlarge,c5.9xlarge,c5.4xlarge"
  },
  {
    "ParameterKey": "OnDemandMaxvCPUs",
    "ParameterValue": "256"
  },
  {
    "ParameterKey": "ECSInstanceProfile",
    "ParameterValue": "${INSTANCE_PROFILE_ARN}"
  },
  {
    "ParameterKey": "BatchServiceRole",
    "ParameterValue": "${BATCH_ROLE_ARN}"
  },
  {
    "ParameterKey": "LaunchTemplateId",
    "ParameterValue": "${LAUNCH_TEMPLATE}"
  },
  {
    "ParameterKey": "SecurityGroupId",
    "ParameterValue": "${SECURITY_GROUP}"
  }
]
EOF
