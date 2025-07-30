#!/bin/bash

# Test runner script for Poppins
echo "Running Poppins Test Suite..."

# Build the test image
echo "Building test image..."
docker build -f Dockerfile.test -t poppins-tests .

if [ $? -ne 0 ]; then
    echo "❌ Failed to build test image"
    exit 1
fi

echo "✅ Test image built successfully"

# Create test results directory
mkdir -p test-results

# Run the tests
echo "Running tests..."
docker run --rm \
    -e TESTING=1 \
    -e OPENAI_API_KEY=test_api_key_for_testing \
    -e OPENAI_MODEL=gpt-4o-realtime-preview \
    -e OPENAI_VOICE=alloy \
    -v "$(pwd)/test-results:/app/test-results" \
    poppins-tests

echo "✅ Tests completed. Check test-results/ for detailed reports."
