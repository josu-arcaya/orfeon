output "instance_server0_ip" {
  description = "Public IP address of the EC2 instance"
  value       = aws_instance.server0.public_ip
}


