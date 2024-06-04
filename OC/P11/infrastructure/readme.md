# Infrastructure

Copy data to s3 bucket

```sh
# Login to Dev account from Infra account (security purpose, no login available directly from dev or prod account)
export DEV_ACCOUNT_ID=267341338450
export INFRA_ACCOUNT_ID=818028758633
export PROD_ACCOUNT_ID=258317103062
export ACCOUNT_ID=$DEV_ACCOUNT_ID

eval $(aws sts assume-role --profile "$INFRA_ACCOUNT_ID" --role-arn "arn:aws:iam::"$ACCOUNT_ID":role/provision" --role-session-name AWSCLI-Session | jq -r '.Credentials | "export AWS_ACCESS_KEY_ID=\(.AccessKeyId)\nexport AWS_SECRET_ACCESS_KEY=\(.SecretAccessKey)\nexport AWS_SESSION_TOKEN=\(.SessionToken)\n"')

# Copy data (only on dev account)
aws s3 cp input/bootstrap-emr.sh s3://267341338450-fruits-oc-data                          
aws s3 cp input/fruits/fruits-360_dataset/fruits-360/Test s3://267341338450-fruits-oc-data/Test --recursive                      
```


Après screenshots

Update S3 policy to allow both profile role
Update security group


```sh
ssh-add ~/.ssh/aws_local_eu.pem 
ssh -D 5555 hadoop@ec2-15-237-100-54.eu-west-3.compute.amazonaws.com

jovyan
jupyter