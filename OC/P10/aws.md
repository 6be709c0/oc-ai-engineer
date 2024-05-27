
```sh
#  The instance used is g4dn.xlarge (0.526 USD per hour) with the AMI: `Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.2.0 (Ubuntu 20.04) 202404101`

export DEV_ACCOUNT_ID=267341338450
export PROD_ACCOUNT_ID=258317103062
export INFRA_ACCOUNT_ID=818028758633
export ACCOUNT_ID=$DEV_ACCOUNT_ID
eval $(aws sts assume-role --profile "$INFRA_ACCOUNT_ID" --role-arn "arn:aws:iam::"$ACCOUNT_ID":role/provision" --role-session-name AWSCLI-Session | jq -r '.Credentials | "export AWS_ACCESS_KEY_ID=\(.AccessKeyId)\nexport AWS_SECRET_ACCESS_KEY=\(.SecretAccessKey)\nexport AWS_SESSION_TOKEN=\(.SessionToken)\n"')

export AWS_REGION=us-east-1
# The name you want to set to your model
export INSTANCE_NAME="ModelTraining"

# The key pair name you have created (here I use a .pem and will add it automatically.
# It should be stored in ~/.ssh/ or modify the code below
export KEY_PAIR="aws_parf_dev"
export SSH_GROUP_NAME="MANUAL_SSHAccess"
# Not need the pager for CLI comfort
export AWS_PAGER=""

SG_GROUP_ID=$(aws ec2 describe-security-groups --query "SecurityGroups[?GroupName=='$SSH_GROUP_NAME'].GroupId" --output text)

#### Supported EC2 instances: G4dn, G5, G6, Gr6, P4, P4de, P5. Release notes: https://docs.aws.amazon.com/dlami/latest/devguide/appendix-ami-release-notes.html
aws ec2 run-instances --image-id ami-04b70fa74e45c3917 \
--instance-type r5.large \
--key-name aws_parf_dev \
--security-group-ids $SG_GROUP_ID \
--tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$INSTANCE_NAME}]"

######
######
######
######

PUBLIC_IP=$(aws ec2 describe-instances \
--filters "Name=tag:Name,Values=$INSTANCE_NAME" "Name=instance-state-name,Values=running" \
--query "Reservations[*].Instances[*].PublicIpAddress" \
--output text)
echo $PUBLIC_IP

ssh-add ~/.ssh/aws_parf_dev.pem
ssh ubuntu@$PUBLIC_IP

mkdir test
exit

scp data.zip ubuntu@$PUBLIC_IP:/home/ubuntu/test/
scp content-based-pca.ipynb ubuntu@$PUBLIC_IP:/home/ubuntu/test/
scp collaborative-filtering.ipynb ubuntu@$PUBLIC_IP:/home/ubuntu/test/
scp helpers.py ubuntu@$PUBLIC_IP:/home/ubuntu/test/
scp requirements.txt ubuntu@$PUBLIC_IP:/home/ubuntu/test/
scp input/lightgcn.yaml ubuntu@$PUBLIC_IP:/home/ubuntu/test/input

# ---
ssh ubuntu@$PUBLIC_IP

sudo apt update
sudo apt install gcc

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

/home/ubuntu/miniconda3/bin/conda init bash
bash

cd test

mkdir -p input/archive
unzip data.zip -d input/archive
mv input/archive/data/* input/archive/
sudo apt-get install unzip -y

conda create -p venv python=3.8 -y
conda activate /home/ubuntu/test/venv
python -m ipykernel install --user --name venv --display-name "conda local"

pip install jupyterlab unzip
pip install -r requirements.txt
jupyter notebook --no-browser >jupyter.log 2>&1 &
ssh -N -f -L 8888:localhost:8888 ubuntu@$PUBLIC_IP
ssh -N -f -L 8887:localhost:8888 ubuntu@34.229.126.140

pkill -f "ssh -N -f -L 8888:localhost:8888"  
```