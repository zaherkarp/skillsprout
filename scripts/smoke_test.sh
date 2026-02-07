#!/bin/bash

# Smoke Test - Quick sanity check that services are working
# Usage: ./scripts/smoke_test.sh [API_URL]

set -e

API_URL="${1:-http://localhost:8000}"
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Running smoke tests against $API_URL${NC}"
echo ""

# Test 1: Health Check
echo -n "1. Health check... "
if curl -sf "$API_URL/api/v1/health" > /dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    exit 1
fi

# Test 2: API Docs
echo -n "2. API docs... "
if curl -sf "$API_URL/api/v1/docs" > /dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    exit 1
fi

# Test 3: Model Status
echo -n "3. Model status... "
if curl -sf "$API_URL/api/v1/model/status" > /dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    exit 1
fi

# Test 4: Search Occupations
echo -n "4. Search occupations... "
if curl -sf "$API_URL/api/v1/occupations/search?q=software" > /dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    exit 1
fi

# Test 5: Create User
echo -n "5. Create user profile... "
response=$(curl -sf -X POST \
    -H "Content-Type: application/json" \
    -d '{}' \
    "$API_URL/api/v1/user/profile")

if echo "$response" | grep -q '"id"'; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}All smoke tests passed!${NC}"
