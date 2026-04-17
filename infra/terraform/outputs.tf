output "eks_cluster_endpoint" {
  description = "EKS cluster API server endpoint"
  value       = module.eks.cluster_endpoint
  sensitive   = false
}

output "eks_cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

output "kafka_bootstrap_brokers" {
  description = "MSK bootstrap broker connection string"
  value       = module.kafka.bootstrap_brokers
  sensitive   = true
}

output "model_store_bucket" {
  description = "S3 bucket for ML model artifacts"
  value       = aws_s3_bucket.model_store.bucket
}

output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}
