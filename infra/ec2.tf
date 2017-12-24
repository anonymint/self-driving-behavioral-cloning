resource "aws_security_group" "ssh" {
  tags = {
    Name = "ssh-sg"
  }

  egress {
    from_port = 0
    to_port = 0
    protocol = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group_rule" "port22" {
  security_group_id = "${aws_security_group.ssh.id}"
  type = "ingress"
  from_port = 22
  to_port = 22
  protocol        = "tcp"
  cidr_blocks     = ["0.0.0.0/0"]
}

resource "aws_security_group_rule" "port8888" {
  security_group_id = "${aws_security_group.ssh.id}"
  type = "ingress"
  from_port = 8888
  to_port = 8888
  protocol        = "tcp"
  cidr_blocks     = ["0.0.0.0/0"]
}

resource "aws_instance" "this" {
  
  ami = "${var.ami_id}" // or equivalent in your region
  instance_type = "${var.instance_type}"
  vpc_security_group_ids = ["${aws_security_group.ssh.id}"]
  associate_public_ip_address = true
 
  tags = {
    Name = "carnd-ami-instance"
  }
}

//resource "null_resource" "carnd_provisioner" {
//  triggers {
//    id = "${uuid()}"
//  }
//
//  connection {
//    host = "${aws_instance.this.public_ip}"
//    user = "carnd"
//    password = "carnd"
//    type = "ssh"
//    agent = true
//  }
//
//
//  provisioner "remote-exec" {
//    inline = [
//      "git clone https://github.com/anonymint/traffic-sign-classifier.git traffic-sign",
//      "cd traffic-sign",
//      "mkdir data_set && wget https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip -O data_set/dataset.zip",
//      "unzip data_set/dataset.zip -d data_set"
//      "source activate carnd-term1",
//      "jupyter notebook --ip=0.0.0.0 --no-browser",
//      ""
//    ]
//  }
//
//
//}

output "instance_ip" {
  value = "${aws_instance.this.public_ip}"
}

