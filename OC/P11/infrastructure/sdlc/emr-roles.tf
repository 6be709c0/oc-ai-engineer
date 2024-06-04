# resource "aws_iam_role" "emr_service_role" {
#   name               = "AmazonEMR-ServiceRole-Terraform"
#   assume_role_policy = data.aws_iam_policy_document.emr_assume_role_polic_service.json
# }

# data "aws_iam_policy_document" "emr_assume_role_polic_service" {
#   statement {
#     actions = ["sts:AssumeRole"]

#     principals {
#       type        = "Service"
#       identifiers = ["elasticmapreduce.amazonaws.com"]
#     }
#   }
# }
# resource "aws_iam_role_policy_attachment" "example" {
#   role       = aws_iam_role.emr_service_role.name
#   policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEMRServicePolicy_v2"
# }
# resource "aws_iam_policy" "emr_service_policy" {
#   name        = "AmazonEMR-ServiceRole-Policy-Terraform"
#   description = "EMR Service policy"

#   policy = jsonencode({
#     "Version" : "2012-10-17",
#     "Statement" : [
#       {
#         "Action" : [
#           "ec2:CreateNetworkInterface",
#           "ec2:RunInstances",
#           "ec2:CreateFleet",
#           "ec2:CreateLaunchTemplate",
#           "ec2:CreateLaunchTemplateVersion"
#         ],
#         "Effect" : "Allow",
#         "Resource" : [
#           "arn:aws:ec2:*:*:subnet/subnet-05d463ca72beb49e7"
#         ],
#         "Sid" : "CreateInNetwork"
#       },
#       {
#         "Action" : [
#           "ec2:CreateSecurityGroup"
#         ],
#         "Effect" : "Allow",
#         "Resource" : [
#           "arn:aws:ec2:*:*:vpc/vpc-0ff5f727b8cd5d9ac"
#         ],
#         "Sid" : "CreateDefaultSecurityGroupInVPC"
#       },
#       {
#         "Sid" : "PassRoleForEC2",
#         "Effect" : "Allow",
#         "Action" : "iam:PassRole",
#         "Resource" : aws_iam_role.emr_instance_profile_role.arn,
#         # "Resource": "arn:aws:iam::267341338450:role/service-role/AmazonEMR-InstanceProfile-20240603T210651",
#         "Condition" : {
#           "StringLike" : {
#             "iam:PassedToService" : "ec2.amazonaws.com"
#           }
#         }
#       }
#     ]
#   })
# }

# resource "aws_iam_role_policy_attachment" "emr_service_policy_attachment" {
#   role       = aws_iam_role.emr_service_role.name
#   policy_arn = aws_iam_policy.emr_service_policy.arn
# }

# resource "aws_iam_role" "emr_instance_profile_role" {
#   name               = "AmazonEMR-InstanceProfile-Terraform"
#   assume_role_policy = data.aws_iam_policy_document.emr_assume_role_policy.json
# }

# resource "aws_iam_instance_profile" "emr_profile" {
#   name = "emr_profile"
#   role = aws_iam_role.emr_instance_profile_role.name
# }

# data "aws_iam_policy_document" "emr_assume_role_policy" {
#   statement {
#     actions = ["sts:AssumeRole"]

#     principals {
#       type        = "Service"
#       identifiers = ["ec2.amazonaws.com"]
#     }
#   }
# }

# resource "aws_iam_policy" "emr_instance_profile_policy" {
#   name        = "AmazonEMR-InstanceProfile-Policy-Terraform"
#   description = "EMR Instance Profile policy"

#   policy = jsonencode({
#     "Version" : "2012-10-17",
#     "Statement" : [
#       {
#         "Effect" : "Allow",
#         "Action" : [
#           "s3:AbortMultipartUpload",
#           "s3:CreateBucket",
#           "s3:DeleteObject",
#           "s3:ListBucket",
#           "s3:ListBucketMultipartUploads",
#           "s3:ListBucketVersions",
#           "s3:ListMultipartUploadParts",
#           "s3:PutBucketVersioning",
#           "s3:PutObject",
#           "s3:PutObjectTagging"
#         ],
#         "Resource" : [
#           "arn:aws:s3:::aws-logs-267341338450-eu-west-3/elasticmapreduce",
#           "arn:aws:s3:::aws-logs-267341338450-eu-west-3/elasticmapreduce/*"
#         ]
#       },
#       {
#         "Effect" : "Allow",
#         "Action" : [
#           "s3:GetBucketVersioning",
#           "s3:GetObject",
#           "s3:GetObjectTagging",
#           "s3:GetObjectVersion",
#           "s3:ListBucket",
#           "s3:ListBucketMultipartUploads",
#           "s3:ListBucketVersions",
#           "s3:ListMultipartUploadParts"
#         ],
#         "Resource" : [
#           "arn:aws:s3:::elasticmapreduce",
#           "arn:aws:s3:::aws-logs-267341338450-eu-west-3/elasticmapreduce",
#           "arn:aws:s3:::elasticmapreduce/*",
#           "arn:aws:s3:::aws-logs-267341338450-eu-west-3/elasticmapreduce/*",
#           "arn:aws:s3:::*.elasticmapreduce/*"
#         ]
#       }
#     ]
#   })
# }

# resource "aws_iam_role_policy_attachment" "emr_instance_profile_policy_attachment" {
#   role       = aws_iam_role.emr_instance_profile_role.name
#   policy_arn = aws_iam_policy.emr_instance_profile_policy.arn
# }
