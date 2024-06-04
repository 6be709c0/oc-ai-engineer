
resource "aws_s3_bucket" "artifacts" {
  bucket = "${data.aws_caller_identity.current.account_id}-fruits-oc-data"

  lifecycle {
    prevent_destroy = true
  }

  tags = {
    Name = "fruits-oc"
  }
}

resource "aws_s3_bucket_policy" "bucket_policy" {
  bucket = aws_s3_bucket.artifacts.id
  
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Principal = {
          AWS =[
            "arn:aws:iam::267341338450:role/service-role/AmazonEMR-InstanceProfile-20240603T210651",
            "arn:aws:iam::267341338450:role/service-role/AmazonEMR-ServiceRole-20240603T210707"
            # aws_iam_role.emr_service_role.arn,
            # aws_iam_role.emr_instance_profile_role.arn
          ]
        },
        Action = [
          "s3:*",
        ],
        Resource = [
          "${aws_s3_bucket.artifacts.arn}",
          "${aws_s3_bucket.artifacts.arn}/*"
        ]
      }
    ]
  })
}