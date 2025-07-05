# Flux PuLID + Depth Deployment Script
Write-Host "================================" -ForegroundColor Cyan
Write-Host "Flux PuLID + Depth Deployment" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Get model name from user
$MODEL_NAME = Read-Host "Enter your Replicate model name (e.g., germz92/flux-pulid-depth)"

if ([string]::IsNullOrWhiteSpace($MODEL_NAME)) {
    Write-Host "Error: Model name is required" -ForegroundColor Red
    pause
    exit 1
}

Write-Host ""
Write-Host "Deploying to: $MODEL_NAME" -ForegroundColor Yellow
Write-Host ""

# Deploy using Docker
Write-Host "Starting deployment using Docker..." -ForegroundColor Green
$deployCommand = "docker run -v `"$PWD`:/src -w /src replicate/cog push r8.im/$MODEL_NAME"
Write-Host "Running: $deployCommand" -ForegroundColor Gray

try {
    Invoke-Expression $deployCommand
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "================================" -ForegroundColor Green
        Write-Host "Deployment successful!" -ForegroundColor Green
        Write-Host "================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "Your model is now available at:" -ForegroundColor Yellow
        Write-Host "https://replicate.com/$MODEL_NAME" -ForegroundColor Blue
        Write-Host ""
    } else {
        Write-Host ""
        Write-Host "================================" -ForegroundColor Red
        Write-Host "Deployment failed!" -ForegroundColor Red
        Write-Host "================================" -ForegroundColor Red
        Write-Host ""
        Write-Host "Common issues:" -ForegroundColor Yellow
        Write-Host "- Make sure you're logged in to Replicate" -ForegroundColor White
        Write-Host "- Check your model name format: username/model-name" -ForegroundColor White
        Write-Host "- Ensure Docker is running" -ForegroundColor White
    }
} catch {
    Write-Host "Error during deployment: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
pause
