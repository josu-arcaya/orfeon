locals {
  cloud_type    = "aws"
  instance_type = "t2.micro"
  ami           = "ami-0ca5c3bd5a268e7db"
  instance_cost = "0.75"
}

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 3.27"
    }
  }
}

provider "aws" {
  profile = "default"
  region  = "us-west-2"
}

resource "aws_key_pair" "deployer" {
  key_name   = "deployer-key"
  public_key = "${file("~/.ssh/id_rsa.pub")}" 
}

resource "aws_instance" "server0" {
  ami           = local.ami 
  instance_type = local.instance_type
  key_name      = "deployer-key"
/*
  provisioner "remote-exec" {
    inline = [
      "sudo sh -c 'echo INSTANCE_COST='${local.instance_cost}' >> /etc/environment'",
      "sudo sh -c 'echo INSTANCE_TYPE='${local.instance_type}' >> /etc/environment'",
      "sudo sh -c 'echo CLOUD_TYPE='${local.cloud_type}' >> /etc/environment'",
      "sudo sh -c 'echo DEBIAN_FRONTEND=noninteractive >> /etc/environment'",
    ]
    connection {
      type        = "ssh"
      host        = self.public_ip
      user        = "ubuntu"
      private_key = "${file("~/.ssh/id_rsa")}"
    }
  }

  provisioner "local-exec" {
    command = "ansible-playbook -u ubuntu -i '${self.public_ip},' ../client.yml"
  }
*/
  tags = {
    Name = "Instance0"
  }
}
