output "ecr_repository_url" {
  description = "ECR repo URL — docker tag/push target"
  value       = aws_ecr_repository.safetyvision.repository_url
}

output "lambda_function_url" {
  description = "Public HTTPS endpoint. POST an image to <url>analyze"
  value       = aws_lambda_function_url.inference_url.function_url
}

output "s3_bucket_name" {
  value = aws_s3_bucket.models.id
}

output "dynamodb_table_name" {
  value = aws_dynamodb_table.violations.name
}
