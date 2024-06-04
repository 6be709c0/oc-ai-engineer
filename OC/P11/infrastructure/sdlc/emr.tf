# aws emr create-cluster \
#  --name "My cluster" \
#  --log-uri "s3://aws-logs-267341338450-eu-west-3/elasticmapreduce" \
#  --release-label "emr-6.3.0" \
#  --service-role "arn:aws:iam::267341338450:role/service-role/AmazonEMR-ServiceRole-20240603T210707" \
#  --unhealthy-node-replacement \
#  --ec2-attributes '{"InstanceProfile":"AmazonEMR-InstanceProfile-20240603T210651","EmrManagedMasterSecurityGroup":"sg-034d248ff370ec90c","EmrManagedSlaveSecurityGroup":"sg-0edba34d08985ac90","KeyName":"aws_local_eu","AdditionalMasterSecurityGroups":[],"AdditionalSlaveSecurityGroups":[],"SubnetId":"subnet-05d463ca72beb49e7"}' \
#  --tags 'for-use-with-amazon-emr-managed-policies=true' \
#  --applications Name=JupyterHub Name=Spark Name=TensorFlow \
#  --configurations '[{"Classification":"jupyter-s3-conf","Properties":{"s3.persistence.bucket":"267341338450-fruits-oc-data","s3.persistence.enabled":"true"}}]' \
#  --instance-groups '[{"InstanceCount":2,"InstanceGroupType":"TASK","Name":"Task - 1","InstanceType":"m5.xlarge","EbsConfiguration":{"EbsBlockDeviceConfigs":[{"VolumeSpecification":{"VolumeType":"gp2","SizeInGB":32},"VolumesPerInstance":2}]}},{"InstanceCount":1,"InstanceGroupType":"MASTER","Name":"Primary","InstanceType":"m5.xlarge","EbsConfiguration":{"EbsBlockDeviceConfigs":[{"VolumeSpecification":{"VolumeType":"gp2","SizeInGB":32},"VolumesPerInstance":2}]}},{"InstanceCount":1,"InstanceGroupType":"CORE","Name":"Core","InstanceType":"m5.xlarge","EbsConfiguration":{"EbsBlockDeviceConfigs":[{"VolumeSpecification":{"VolumeType":"gp2","SizeInGB":32},"VolumesPerInstance":2}]}}]' \
#  --bootstrap-actions '[{"Args":[],"Name":"bootstrap-emr","Path":"s3://267341338450-fruits-oc-data/bootstrap-emr.sh"}]' \
#  --scale-down-behavior "TERMINATE_AT_TASK_COMPLETION" \
#  --auto-termination-policy '{"IdleTimeout":3600}' \
#  --region "eu-west-3"



# resource "aws_emr_cluster" "my_cluster" {
#   name                = "My cluster"
#   release_label       = "emr-6.3.0"
#   log_uri             = "s3://aws-logs-267341338450-eu-west-3/elasticmapreduce"
#   service_role        = aws_iam_role.emr_service_role.arn
#   scale_down_behavior = "TERMINATE_AT_TASK_COMPLETION"
#   auto_termination_policy { idle_timeout = 7200 }

#   applications = ["JupyterHub", "Spark", "TensorFlow"]

#   tags = {
#     "for-use-with-amazon-emr-managed-policies" = "true"
#   }

#   ec2_attributes {
#     instance_profile                  = aws_iam_instance_profile.emr_profile.arn
#     emr_managed_master_security_group = "sg-034d248ff370ec90c"
#     emr_managed_slave_security_group  = "sg-0edba34d08985ac90"
#     subnet_id                         = "subnet-05d463ca72beb49e7"
#     key_name                          = "aws_local_eu"
#   }

#   configurations_json = <<EOF
# [
#   {
#     "Classification": "jupyter-s3-conf",
#     "Properties": {
#       "s3.persistence.bucket": "${aws_s3_bucket.artifacts.id}",
#       "s3.persistence.enabled": "true"
#     }
#   }
# ]
# EOF

#   bootstrap_action {
#     name = "bootstrap-emr"
#     path = "s3://${aws_s3_bucket.artifacts.id}/bootstrap-emr.sh"
#   }

#   master_instance_group {
#     instance_type  = "m5.xlarge"
#     instance_count = 1

#     ebs_config {
#       size                 = 32
#       type                 = "gp2"
#       volumes_per_instance = 2
#     }
#   }

#   core_instance_group {
#     instance_type  = "m5.xlarge"
#     instance_count = 1

#     ebs_config {
#       size                 = 32
#       type                 = "gp2"
#       volumes_per_instance = 2
#     }
#   }

#   # step {
#   #   name              = "Unhealthy node replacement"
#   #   action_on_failure = "TERMINATE_CLUSTER"
#   # }
# }
# resource "aws_emr_instance_group" "task" {
#   cluster_id     = aws_emr_cluster.my_cluster.id
#   instance_count = 2
#   instance_type  = "m5.xlarge"
#   name           = "task instance group"
# }