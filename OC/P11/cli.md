```sh
aws emr create-cluster \
 --name "My cluster" \
 --log-uri "s3://aws-logs-267341338450-eu-west-3/elasticmapreduce" \
 --release-label "emr-7.0.0" \
 --service-role "arn:aws:iam::267341338450:role/service-role/AmazonEMR-ServiceRole-20240603T210707" \
 --unhealthy-node-replacement \
 --ec2-attributes '{"InstanceProfile":"AmazonEMR-InstanceProfile-20240603T210651","EmrManagedMasterSecurityGroup":"sg-034d248ff370ec90c","EmrManagedSlaveSecurityGroup":"sg-0edba34d08985ac90","KeyName":"aws_local_eu","AdditionalMasterSecurityGroups":[],"AdditionalSlaveSecurityGroups":[],"SubnetId":"subnet-05d463ca72beb49e7"}' \
 --tags 'for-use-with-amazon-emr-managed-policies=true' \
 --applications Name=Hadoop Name=JupyterHub Name=Spark \
 --configurations '[{"Classification":"jupyter-s3-conf","Properties":{"s3.persistence.bucket":"267341338450-fruits-oc-data","s3.persistence.enabled":"true"}}]' \
 --instance-groups '[{"InstanceCount":1,"InstanceGroupType":"CORE","Name":"Core","InstanceType":"m5.xlarge","EbsConfiguration":{"EbsBlockDeviceConfigs":[{"VolumeSpecification":{"VolumeType":"gp2","SizeInGB":32},"VolumesPerInstance":2}]}},{"InstanceCount":2,"InstanceGroupType":"TASK","Name":"Task - 1","InstanceType":"m5.xlarge","EbsConfiguration":{"EbsBlockDeviceConfigs":[{"VolumeSpecification":{"VolumeType":"gp2","SizeInGB":32},"VolumesPerInstance":2}]}},{"InstanceCount":1,"InstanceGroupType":"MASTER","Name":"Primary","InstanceType":"m5.xlarge","EbsConfiguration":{"EbsBlockDeviceConfigs":[{"VolumeSpecification":{"VolumeType":"gp2","SizeInGB":32},"VolumesPerInstance":2}]}}]' \
 --bootstrap-actions '[{"Args":[],"Name":"bootstrap-emr","Path":"s3://267341338450-fruits-oc-data/bootstrap-emr.sh"}]' \
 --scale-down-behavior "TERMINATE_AT_TASK_COMPLETION" \
 --auto-termination-policy '{"IdleTimeout":3600}' \
 --region "eu-west-3"