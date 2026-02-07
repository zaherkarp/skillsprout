#!/bin/bash

# SkillSprout End-to-End Test
# Tests the full user workflow via API

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

API_URL="${API_URL:-http://localhost:8000}"
PASSED=0
FAILED=0

echo -e "${BLUE}"
echo "========================================"
echo "  SkillSprout E2E Test"
echo "========================================"
echo -e "${NC}"
echo "API URL: $API_URL"
echo ""

# Helper: extract a JSON field using python3 (always available on ubuntu-latest)
json_field() {
    python3 -c "import sys,json; data=json.loads(sys.stdin.read()); print(json.dumps(data$1))" 2>/dev/null | tr -d '"'
}

json_array_item() {
    python3 -c "import sys,json; data=json.loads(sys.stdin.read()); print(data$1)" 2>/dev/null
}

# Function to call an endpoint, print status, and store the raw JSON body in $BODY
call_endpoint() {
    local name="$1"
    local method="$2"
    local endpoint="$3"
    local data="$4"
    local expected_code="${5:-200}"

    echo -n "  Testing: $name... "

    if [ "$method" == "GET" ]; then
        response=$(curl -s -w "\n%{http_code}" "$API_URL$endpoint")
    else
        response=$(curl -s -w "\n%{http_code}" -X "$method" \
            -H "Content-Type: application/json" \
            -d "$data" \
            "$API_URL$endpoint")
    fi

    http_code=$(echo "$response" | tail -n1)
    BODY=$(echo "$response" | sed '$d')

    if [ "$http_code" == "$expected_code" ]; then
        echo -e "${GREEN}✓${NC}"
        PASSED=$((PASSED + 1))
        return 0
    else
        echo -e "${RED}✗ (Expected $expected_code, got $http_code)${NC}"
        echo "  Response: $BODY"
        FAILED=$((FAILED + 1))
        return 1
    fi
}

# Wait for API to be ready
echo -e "${YELLOW}Waiting for API to be ready...${NC}"
max_attempts=30
attempt=0
while [ $attempt -lt $max_attempts ]; do
    if curl -s "$API_URL/api/v1/health" > /dev/null 2>&1; then
        echo -e "${GREEN}API is ready!${NC}"
        break
    fi
    attempt=$((attempt + 1))
    sleep 2
done

if [ $attempt -eq $max_attempts ]; then
    echo -e "${RED}API did not become ready in time${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}Running E2E Tests...${NC}"
echo ""

# 1. Health Check
call_endpoint "Health Check" "GET" "/api/v1/health" "" 200

# 2. Create User Profile
echo -e "\n${YELLOW}Step 1: Create User Profile${NC}"
call_endpoint "Create User" "POST" "/api/v1/user/profile" '{}' 201
user_id=$(echo "$BODY" | json_field '["id"]')
echo "  User ID: $user_id"

if [ -z "$user_id" ]; then
    echo -e "${RED}Failed to extract user ID${NC}"
    exit 1
fi

# 3. Search Occupations
echo -e "\n${YELLOW}Step 2: Search Occupations${NC}"
call_endpoint "Search Occupations" "GET" "/api/v1/occupations/search?q=software" "" 200
onet_code=$(echo "$BODY" | python3 -c "import sys,json; data=json.loads(sys.stdin.read()); print(data[0]['code'])" 2>/dev/null)
echo "  O*NET Code: $onet_code"

if [ -z "$onet_code" ]; then
    echo -e "${RED}Failed to extract O*NET code${NC}"
    exit 1
fi

# 4. Get Occupation Details
echo -e "\n${YELLOW}Step 3: Get Occupation Details${NC}"
call_endpoint "Get Occupation" "GET" "/api/v1/occupations/$onet_code" "" 200

# 5. Get Occupation Skills
echo -e "\n${YELLOW}Step 4: Get Occupation Skills${NC}"
call_endpoint "Get Skills" "GET" "/api/v1/occupations/$onet_code/skills" "" 200
skill1=$(echo "$BODY" | python3 -c "import sys,json; data=json.loads(sys.stdin.read()); print(data['skills'][0]['element_id'])" 2>/dev/null)
skill2=$(echo "$BODY" | python3 -c "import sys,json; data=json.loads(sys.stdin.read()); print(data['skills'][1]['element_id'])" 2>/dev/null)
skill3=$(echo "$BODY" | python3 -c "import sys,json; data=json.loads(sys.stdin.read()); print(data['skills'][2]['element_id'])" 2>/dev/null)
echo "  Skills found: $skill1, $skill2, $skill3"

if [ -z "$skill1" ] || [ -z "$skill2" ] || [ -z "$skill3" ]; then
    echo -e "${RED}Failed to extract skill element IDs from response${NC}"
    echo "  Response body: $BODY"
    exit 1
fi

# 6. Set Current Occupation
echo -e "\n${YELLOW}Step 5: Set Current Occupation${NC}"
call_endpoint "Set Occupation" "POST" "/api/v1/user/$user_id/current-occupation" \
    "{\"onet_code\": \"$onet_code\"}" 200

# 7. Rate Skills
echo -e "\n${YELLOW}Step 6: Rate Skills${NC}"
ratings_data="{
    \"ratings\": [
        {\"element_id\": \"$skill1\", \"rating_0_4\": 4},
        {\"element_id\": \"$skill2\", \"rating_0_4\": 3},
        {\"element_id\": \"$skill3\", \"rating_0_4\": 2}
    ]
}"
call_endpoint "Rate Skills" "POST" "/api/v1/user/$user_id/skills/ratings" \
    "$ratings_data" 200

# 8. Get Recommendations
echo -e "\n${YELLOW}Step 7: Get Recommendations${NC}"
call_endpoint "Get Recommendations" "POST" "/api/v1/user/$user_id/recommendations" \
    '{"limit_per_bucket": 5}' 200
event_id=$(echo "$BODY" | json_field '["event_id"]')
target_code=$(echo "$BODY" | python3 -c "
import sys,json
data=json.loads(sys.stdin.read())
for b in data.get('buckets',[]):
    for o in b.get('occupations',[]):
        if o['onet_code'] != '$onet_code':
            print(o['onet_code'])
            sys.exit(0)
print('')
" 2>/dev/null)
echo "  Event ID: $event_id"
echo "  Sample Recommendation: $target_code"

# 9. Submit Feedback
if [ -n "$target_code" ] && [ -n "$event_id" ]; then
    echo -e "\n${YELLOW}Step 8: Submit Feedback${NC}"
    call_endpoint "Submit Feedback" "POST" "/api/v1/feedback" \
        "{\"event_id\": $event_id, \"target_onet_code\": \"$target_code\", \"action_type\": \"click\"}" \
        201
fi

# 10. Check Model Status
echo -e "\n${YELLOW}Step 9: Check Model Status${NC}"
call_endpoint "Model Status" "GET" "/api/v1/model/status" "" 200

# Results
echo ""
echo -e "${BLUE}========================================"
echo "  Test Results"
echo "========================================${NC}"
echo -e "Passed: ${GREEN}$PASSED${NC}"
echo -e "Failed: ${RED}$FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All E2E tests passed!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some E2E tests failed${NC}"
    exit 1
fi
