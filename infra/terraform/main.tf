terraform {
  required_version = ">= 1.7"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.25"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.12"
    }
  }
  backend "s3" {
    bucket         = "fraudgraphx-tf-state"
    key            = "prod/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "fraudgraphx-tf-locks"
  }
}

provider "aws" {
  region = var.aws_region
  default_tags {
    tags = {
      Project     = "FraudGraphX"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# ── VPC ───────────────────────────────────────────────────────────────────────
module "vpc" {
  source = "./modules/vpc"

  vpc_name        = "fraudgraphx-${var.environment}"
  vpc_cidr        = var.vpc_cidr
  azs             = var.availability_zones
  private_subnets = var.private_subnet_cidrs
  public_subnets  = var.public_subnet_cidrs
}

# ── EKS ───────────────────────────────────────────────────────────────────────
module "eks" {
  source = "./modules/eks"

  cluster_name    = "fraudgraphx-${var.environment}"
  cluster_version = "1.29"
  vpc_id          = module.vpc.vpc_id
  subnet_ids      = module.vpc.private_subnet_ids

  node_groups = {
    api = {
      instance_types = ["m6i.xlarge"]
      min_size       = 2
      max_size       = 10
      desired_size   = 3
    }
    ml = {
      instance_types = ["g4dn.xlarge"]  # GPU for GNN inference
      min_size       = 1
      max_size       = 5
      desired_size   = 2
    }
    streaming = {
      instance_types = ["r6i.2xlarge"]
      min_size       = 2
      max_size       = 8
      desired_size   = 3
    }
  }
}

# ── MSK (Kafka) ───────────────────────────────────────────────────────────────
module "kafka" {
  source = "./modules/kafka"

  cluster_name      = "fraudgraphx-${var.environment}"
  vpc_id            = module.vpc.vpc_id
  subnet_ids        = module.vpc.private_subnet_ids
  kafka_version     = "3.5.1"
  broker_node_type  = "kafka.m5.2xlarge"
  broker_count      = 3
  ebs_volume_size   = 500
}

# ── Cassandra (Amazon Keyspaces) ──────────────────────────────────────────────
module "cassandra" {
  source = "./modules/cassandra"

  keyspace_name = "fraudgraphx"
  environment   = var.environment
}

# ── Neo4j (EC2 cluster) ───────────────────────────────────────────────────────
module "neo4j" {
  source = "./modules/neo4j"

  vpc_id        = module.vpc.vpc_id
  subnet_ids    = module.vpc.private_subnet_ids
  instance_type = "r6i.2xlarge"
  cluster_size  = 3
  environment   = var.environment
}

# ── S3 for model artifacts ────────────────────────────────────────────────────
resource "aws_s3_bucket" "model_store" {
  bucket = "fraudgraphx-models-${var.environment}-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket_versioning" "model_store" {
  bucket = aws_s3_bucket.model_store.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "model_store" {
  bucket = aws_s3_bucket.model_store.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
  }
}

data "aws_caller_identity" "current" {}

# ── CloudWatch Alarms ─────────────────────────────────────────────────────────
resource "aws_cloudwatch_metric_alarm" "fraud_alert_latency" {
  alarm_name          = "fraudgraphx-alert-latency-breach"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "fraudgraphx_request_latency_ms"
  namespace           = "FraudGraphX"
  period              = 60
  statistic           = "p99"
  threshold           = 150
  alarm_description   = "Alert latency SLA breach (>150ms p99)"
  alarm_actions       = [aws_sns_topic.alerts.arn]
}

resource "aws_sns_topic" "alerts" {
  name = "fraudgraphx-${var.environment}-ops-alerts"
}
