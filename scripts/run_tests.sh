#!/bin/bash

# SkillSprout Test Runner
# Runs the full test suite in an isolated Docker environment

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "========================================"
echo "  SkillSprout Test Suite"
echo "========================================"
echo -e "${NC}"

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Cleaning up test environment...${NC}"
    docker compose -f docker-compose.test.yml down -v > /dev/null 2>&1 || true
}

# Set trap to cleanup on exit
trap cleanup EXIT

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running${NC}"
    exit 1
fi

# Build test image if needed
echo -e "${YELLOW}Building test image...${NC}"
docker compose -f docker-compose.test.yml build

# Run tests
echo -e "${GREEN}Starting test environment...${NC}"
docker compose -f docker-compose.test.yml up --abort-on-container-exit --exit-code-from test-runner

# Capture exit code
EXIT_CODE=$?

# Print results
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}"
    echo "========================================"
    echo "  ✓ All Tests Passed!"
    echo "========================================"
    echo -e "${NC}"
else
    echo -e "${RED}"
    echo "========================================"
    echo "  ✗ Tests Failed"
    echo "========================================"
    echo -e "${NC}"
fi

# Copy coverage report if it exists
if docker volume inspect skillsprout_test_coverage > /dev/null 2>&1; then
    echo -e "${YELLOW}Extracting coverage report...${NC}"
    CONTAINER_ID=$(docker create -v skillsprout_test_coverage:/data busybox)
    docker cp $CONTAINER_ID:/data/. ./htmlcov/ 2>/dev/null || true
    docker rm $CONTAINER_ID > /dev/null 2>&1 || true
    if [ -d "./htmlcov" ]; then
        echo -e "${GREEN}Coverage report available at: htmlcov/index.html${NC}"
    fi
fi

exit $EXIT_CODE
