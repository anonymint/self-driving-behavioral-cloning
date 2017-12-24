variable "ami_id" {
  description = "AMI ID for create EC2"
}

variable "aws_region" {
  description = "region to create EC2 in"
}

variable "instance_type" {
  description = "Instance type of EC2"
  default = "p2.xlarge"
}
