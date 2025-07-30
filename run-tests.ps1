# PowerShell test runner script for Poppins
Write-Host "Running Poppins Test Suite..." -ForegroundColor Green

# Build the test image
Write-Host "Building test image..." -ForegroundColor Yellow
docker build -f Dockerfile.test -t poppins-tests .

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to build test image" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Test image built successfully" -ForegroundColor Green

# Create test results directory
if (!(Test-Path "test-results")) {
    New-Item -ItemType Directory -Path "test-results"
}

# Run the tests
Write-Host "Running tests..." -ForegroundColor Yellow
docker run --rm `
    -e TESTING=1 `
    -e OPENAI_API_KEY=test_api_key_for_testing `
    -e OPENAI_MODEL=gpt-4o-realtime-preview `
    -e OPENAI_VOICE=alloy `
    -v "${PWD}/test-results:/app/test-results" `
    poppins-tests

Write-Host "✅ Tests completed. Check test-results/ for detailed reports." -ForegroundColor Green
