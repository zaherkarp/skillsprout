"""Comprehensive tests for 100 professions across all 22 SOC sectors.

Tests the BaselineScorer against realistic occupation data spanning
every major occupational sector, validating scoring invariants,
cross-sector transitions, and bucket assignment correctness.
"""
import pytest
from typing import Dict, List, Any

from app.ml.scoring import BaselineScorer, OccupationScore


# ============================================================================
# Skill element_id reference
# ============================================================================

# ============================================================================
# 100 professions across all 22 SOC sectors
# Each entry: onet_code -> {title, job_zone, sector, skills}
# ============================================================================

PROFESSIONS_100 = {
    "11-1021.00": {
        "title": "General and Operations Managers",
        "job_zone": 4,
        "sector": "Management",
        "skills": [
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 75, "level": 5.25},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 78, "level": 5.38},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 81, "level": 5.62},
            {"element_id": "2.B.4.e", "skill_name": "Management of Personnel Resources", "importance": 84, "level": 5.75},
            {"element_id": "2.B.4.f", "skill_name": "Management of Financial Resources", "importance": 75, "level": 5.12},
            {"element_id": "2.B.7.b", "skill_name": "Coordination", "importance": 78, "level": 5.38},
            {"element_id": "2.B.3.c", "skill_name": "Negotiation", "importance": 72, "level": 5.0},
            {"element_id": "2.B.6.b", "skill_name": "Time Management", "importance": 75, "level": 5.25},
        ],
    },
    "11-2021.00": {
        "title": "Marketing Managers",
        "job_zone": 4,
        "sector": "Management",
        "skills": [
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 72, "level": 5.12},
            {"element_id": "2.B.3.a", "skill_name": "Writing", "importance": 78, "level": 5.38},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 81, "level": 5.62},
            {"element_id": "2.B.3.b", "skill_name": "Persuasion", "importance": 84, "level": 5.75},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 78, "level": 5.38},
            {"element_id": "2.B.7.a", "skill_name": "Social Perceptiveness", "importance": 75, "level": 5.25},
            {"element_id": "2.B.6.b", "skill_name": "Time Management", "importance": 72, "level": 5.0},
            {"element_id": "2.B.4.f", "skill_name": "Management of Financial Resources", "importance": 69, "level": 4.88},
        ],
    },
    "11-3021.00": {
        "title": "Computer and Information Systems Managers",
        "job_zone": 5,
        "sector": "Management",
        "skills": [
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 75, "level": 5.25},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 84, "level": 5.88},
            {"element_id": "2.B.8.b", "skill_name": "Complex Problem Solving", "importance": 81, "level": 5.62},
            {"element_id": "2.B.8.d", "skill_name": "Systems Analysis", "importance": 78, "level": 5.38},
            {"element_id": "2.B.4.e", "skill_name": "Management of Personnel Resources", "importance": 78, "level": 5.38},
            {"element_id": "2.B.1.g", "skill_name": "Programming", "importance": 69, "level": 4.88},
            {"element_id": "2.B.6.b", "skill_name": "Time Management", "importance": 75, "level": 5.25},
            {"element_id": "2.B.7.b", "skill_name": "Coordination", "importance": 75, "level": 5.25},
        ],
    },
    "11-9013.00": {
        "title": "Farmers, Ranchers, and Agricultural Managers",
        "job_zone": 3,
        "sector": "Management",
        "skills": [
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 72, "level": 5.0},
            {"element_id": "2.B.8.b", "skill_name": "Complex Problem Solving", "importance": 69, "level": 4.88},
            {"element_id": "2.B.4.g", "skill_name": "Management of Material Resources", "importance": 78, "level": 5.38},
            {"element_id": "2.B.4.f", "skill_name": "Management of Financial Resources", "importance": 72, "level": 5.0},
            {"element_id": "2.B.7.b", "skill_name": "Coordination", "importance": 69, "level": 4.88},
            {"element_id": "2.B.9.c", "skill_name": "Operations Monitoring", "importance": 75, "level": 5.25},
            {"element_id": "2.B.6.b", "skill_name": "Time Management", "importance": 72, "level": 5.0},
        ],
    },
    "13-1111.00": {
        "title": "Management Analysts",
        "job_zone": 4,
        "sector": "Financial",
        "skills": [
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 78, "level": 5.38},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 84, "level": 5.88},
            {"element_id": "2.B.3.a", "skill_name": "Writing", "importance": 78, "level": 5.38},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 75, "level": 5.25},
            {"element_id": "2.B.8.b", "skill_name": "Complex Problem Solving", "importance": 78, "level": 5.38},
            {"element_id": "2.B.8.d", "skill_name": "Systems Analysis", "importance": 75, "level": 5.25},
            {"element_id": "2.B.8.e", "skill_name": "Systems Evaluation", "importance": 72, "level": 5.0},
            {"element_id": "2.B.6.b", "skill_name": "Time Management", "importance": 69, "level": 4.88},
        ],
    },
    "13-2011.00": {
        "title": "Accountants and Auditors",
        "job_zone": 4,
        "sector": "Financial",
        "skills": [
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 75, "level": 5.25},
            {"element_id": "2.B.5.a", "skill_name": "Mathematics", "importance": 84, "level": 5.88},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 78, "level": 5.38},
            {"element_id": "2.B.3.a", "skill_name": "Writing", "importance": 72, "level": 5.0},
            {"element_id": "2.B.8.d", "skill_name": "Systems Analysis", "importance": 69, "level": 4.88},
            {"element_id": "2.B.6.b", "skill_name": "Time Management", "importance": 72, "level": 5.0},
            {"element_id": "2.B.7.c", "skill_name": "Monitoring", "importance": 69, "level": 4.88},
        ],
    },
    "13-1161.00": {
        "title": "Market Research Analysts",
        "job_zone": 4,
        "sector": "Financial",
        "skills": [
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 78, "level": 5.38},
            {"element_id": "2.B.5.a", "skill_name": "Mathematics", "importance": 75, "level": 5.25},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 81, "level": 5.62},
            {"element_id": "2.B.3.a", "skill_name": "Writing", "importance": 75, "level": 5.25},
            {"element_id": "2.B.8.b", "skill_name": "Complex Problem Solving", "importance": 72, "level": 5.0},
            {"element_id": "2.B.8.d", "skill_name": "Systems Analysis", "importance": 69, "level": 4.88},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 66, "level": 4.62},
        ],
    },
    "13-2051.00": {
        "title": "Financial Analysts",
        "job_zone": 4,
        "sector": "Financial",
        "skills": [
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 78, "level": 5.38},
            {"element_id": "2.B.5.a", "skill_name": "Mathematics", "importance": 87, "level": 6.0},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 84, "level": 5.88},
            {"element_id": "2.B.8.b", "skill_name": "Complex Problem Solving", "importance": 78, "level": 5.38},
            {"element_id": "2.B.8.d", "skill_name": "Systems Analysis", "importance": 75, "level": 5.25},
            {"element_id": "2.B.3.a", "skill_name": "Writing", "importance": 72, "level": 5.0},
            {"element_id": "2.B.6.b", "skill_name": "Time Management", "importance": 69, "level": 4.88},
        ],
    },
    "15-1252.00": {
        "title": "Software Developers",
        "job_zone": 4,
        "sector": "Information/Tech",
        "skills": [
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 72, "level": 5.12},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 81, "level": 5.62},
            {"element_id": "2.B.8.b", "skill_name": "Complex Problem Solving", "importance": 84, "level": 5.88},
            {"element_id": "2.B.8.d", "skill_name": "Systems Analysis", "importance": 78, "level": 5.38},
            {"element_id": "2.B.1.g", "skill_name": "Programming", "importance": 84, "level": 5.75},
            {"element_id": "2.B.3.a", "skill_name": "Writing", "importance": 75, "level": 5.12},
            {"element_id": "2.B.5.a", "skill_name": "Mathematics", "importance": 66, "level": 4.62},
            {"element_id": "2.B.6.b", "skill_name": "Time Management", "importance": 69, "level": 4.88},
        ],
    },
    "15-1299.08": {
        "title": "Web Developers",
        "job_zone": 3,
        "sector": "Information/Tech",
        "skills": [
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 69, "level": 4.88},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 75, "level": 5.25},
            {"element_id": "2.B.8.b", "skill_name": "Complex Problem Solving", "importance": 78, "level": 5.38},
            {"element_id": "2.B.1.g", "skill_name": "Programming", "importance": 81, "level": 5.62},
            {"element_id": "2.B.5.c", "skill_name": "Design", "importance": 72, "level": 5.0},
            {"element_id": "2.B.6.b", "skill_name": "Time Management", "importance": 66, "level": 4.75},
        ],
    },
    "15-1212.00": {
        "title": "Information Security Analysts",
        "job_zone": 4,
        "sector": "Information/Tech",
        "skills": [
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 75, "level": 5.25},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 84, "level": 5.88},
            {"element_id": "2.B.8.b", "skill_name": "Complex Problem Solving", "importance": 81, "level": 5.62},
            {"element_id": "2.B.8.d", "skill_name": "Systems Analysis", "importance": 78, "level": 5.38},
            {"element_id": "2.B.1.g", "skill_name": "Programming", "importance": 72, "level": 5.0},
            {"element_id": "2.B.9.a", "skill_name": "Troubleshooting", "importance": 75, "level": 5.25},
            {"element_id": "2.B.7.c", "skill_name": "Monitoring", "importance": 72, "level": 5.0},
        ],
    },
    "15-1244.00": {
        "title": "Network and Computer Systems Administrators",
        "job_zone": 3,
        "sector": "Information/Tech",
        "skills": [
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 69, "level": 5.0},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 75, "level": 5.25},
            {"element_id": "2.B.8.b", "skill_name": "Complex Problem Solving", "importance": 81, "level": 5.5},
            {"element_id": "2.B.9.a", "skill_name": "Troubleshooting", "importance": 84, "level": 5.75},
            {"element_id": "2.B.1.g", "skill_name": "Programming", "importance": 72, "level": 5.12},
            {"element_id": "2.B.8.d", "skill_name": "Systems Analysis", "importance": 78, "level": 5.38},
            {"element_id": "2.B.4.h", "skill_name": "Equipment Maintenance", "importance": 66, "level": 4.5},
        ],
    },
    "15-1241.00": {
        "title": "Computer Network Architects",
        "job_zone": 4,
        "sector": "Information/Tech",
        "skills": [
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 72, "level": 5.12},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 81, "level": 5.62},
            {"element_id": "2.B.8.b", "skill_name": "Complex Problem Solving", "importance": 84, "level": 5.88},
            {"element_id": "2.B.8.d", "skill_name": "Systems Analysis", "importance": 81, "level": 5.62},
            {"element_id": "2.B.8.e", "skill_name": "Systems Evaluation", "importance": 78, "level": 5.38},
            {"element_id": "2.B.1.g", "skill_name": "Programming", "importance": 75, "level": 5.25},
            {"element_id": "2.B.5.c", "skill_name": "Design", "importance": 69, "level": 4.88},
        ],
    },
    "17-2051.00": {
        "title": "Civil Engineers",
        "job_zone": 4,
        "sector": "Engineering/Construction",
        "skills": [
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 75, "level": 5.25},
            {"element_id": "2.B.5.a", "skill_name": "Mathematics", "importance": 84, "level": 5.88},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 81, "level": 5.62},
            {"element_id": "2.B.8.b", "skill_name": "Complex Problem Solving", "importance": 78, "level": 5.38},
            {"element_id": "2.B.8.d", "skill_name": "Systems Analysis", "importance": 75, "level": 5.25},
            {"element_id": "2.B.3.a", "skill_name": "Writing", "importance": 72, "level": 5.0},
            {"element_id": "2.B.5.c", "skill_name": "Design", "importance": 69, "level": 4.88},
            {"element_id": "2.B.5.b", "skill_name": "Science", "importance": 66, "level": 4.62},
        ],
    },
    "17-2141.00": {
        "title": "Mechanical Engineers",
        "job_zone": 4,
        "sector": "Engineering/Construction",
        "skills": [
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 75, "level": 5.25},
            {"element_id": "2.B.5.a", "skill_name": "Mathematics", "importance": 81, "level": 5.62},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 81, "level": 5.62},
            {"element_id": "2.B.8.b", "skill_name": "Complex Problem Solving", "importance": 81, "level": 5.62},
            {"element_id": "2.B.5.c", "skill_name": "Design", "importance": 78, "level": 5.38},
            {"element_id": "2.B.8.d", "skill_name": "Systems Analysis", "importance": 75, "level": 5.25},
            {"element_id": "2.B.5.b", "skill_name": "Science", "importance": 69, "level": 4.88},
        ],
    },
    "17-3023.00": {
        "title": "Electrical and Electronic Engineering Technologists",
        "job_zone": 3,
        "sector": "Engineering/Construction",
        "skills": [
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 72, "level": 5.0},
            {"element_id": "2.B.5.a", "skill_name": "Mathematics", "importance": 75, "level": 5.25},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 75, "level": 5.25},
            {"element_id": "2.B.9.a", "skill_name": "Troubleshooting", "importance": 78, "level": 5.38},
            {"element_id": "2.B.9.c", "skill_name": "Operations Monitoring", "importance": 72, "level": 5.0},
            {"element_id": "2.B.6.c", "skill_name": "Quality Control Analysis", "importance": 69, "level": 4.88},
        ],
    },
    "17-2071.00": {
        "title": "Electrical Engineers",
        "job_zone": 4,
        "sector": "Engineering/Construction",
        "skills": [
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 75, "level": 5.25},
            {"element_id": "2.B.5.a", "skill_name": "Mathematics", "importance": 84, "level": 5.88},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 84, "level": 5.88},
            {"element_id": "2.B.8.b", "skill_name": "Complex Problem Solving", "importance": 81, "level": 5.62},
            {"element_id": "2.B.5.c", "skill_name": "Design", "importance": 78, "level": 5.38},
            {"element_id": "2.B.8.d", "skill_name": "Systems Analysis", "importance": 78, "level": 5.38},
            {"element_id": "2.B.5.b", "skill_name": "Science", "importance": 72, "level": 5.0},
        ],
    },
    "19-1042.00": {
        "title": "Medical Scientists",
        "job_zone": 5,
        "sector": "Science/Professional",
        "skills": [
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 81, "level": 5.62},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 84, "level": 5.88},
            {"element_id": "2.B.5.b", "skill_name": "Science", "importance": 87, "level": 6.0},
            {"element_id": "2.B.3.a", "skill_name": "Writing", "importance": 78, "level": 5.38},
            {"element_id": "2.B.8.b", "skill_name": "Complex Problem Solving", "importance": 78, "level": 5.38},
            {"element_id": "2.B.5.a", "skill_name": "Mathematics", "importance": 72, "level": 5.0},
            {"element_id": "2.B.8.d", "skill_name": "Systems Analysis", "importance": 69, "level": 4.88},
        ],
    },
    "19-2041.00": {
        "title": "Environmental Scientists",
        "job_zone": 4,
        "sector": "Science/Professional",
        "skills": [
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 78, "level": 5.38},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 81, "level": 5.62},
            {"element_id": "2.B.5.b", "skill_name": "Science", "importance": 84, "level": 5.88},
            {"element_id": "2.B.3.a", "skill_name": "Writing", "importance": 75, "level": 5.25},
            {"element_id": "2.B.8.b", "skill_name": "Complex Problem Solving", "importance": 72, "level": 5.0},
            {"element_id": "2.B.5.a", "skill_name": "Mathematics", "importance": 69, "level": 4.88},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 66, "level": 4.62},
        ],
    },
    "19-4099.00": {
        "title": "Life, Physical, and Social Science Technicians",
        "job_zone": 3,
        "sector": "Science/Professional",
        "skills": [
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 69, "level": 4.88},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 72, "level": 5.0},
            {"element_id": "2.B.5.b", "skill_name": "Science", "importance": 75, "level": 5.25},
            {"element_id": "2.B.6.c", "skill_name": "Quality Control Analysis", "importance": 72, "level": 5.0},
            {"element_id": "2.B.9.c", "skill_name": "Operations Monitoring", "importance": 69, "level": 4.88},
            {"element_id": "2.B.5.a", "skill_name": "Mathematics", "importance": 66, "level": 4.62},
        ],
    },
    "19-3031.00": {
        "title": "Clinical and Counseling Psychologists",
        "job_zone": 5,
        "sector": "Science/Professional",
        "skills": [
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 78, "level": 5.38},
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 84, "level": 5.88},
            {"element_id": "2.B.7.a", "skill_name": "Social Perceptiveness", "importance": 87, "level": 6.0},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 81, "level": 5.62},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 78, "level": 5.38},
            {"element_id": "2.B.3.a", "skill_name": "Writing", "importance": 75, "level": 5.25},
            {"element_id": "2.B.5.b", "skill_name": "Science", "importance": 69, "level": 4.88},
        ],
    },
    "19-1031.00": {
        "title": "Conservation Scientists",
        "job_zone": 4,
        "sector": "Science/Professional",
        "skills": [
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 75, "level": 5.25},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 78, "level": 5.38},
            {"element_id": "2.B.5.b", "skill_name": "Science", "importance": 78, "level": 5.38},
            {"element_id": "2.B.3.a", "skill_name": "Writing", "importance": 72, "level": 5.0},
            {"element_id": "2.B.8.b", "skill_name": "Complex Problem Solving", "importance": 69, "level": 4.88},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 66, "level": 4.62},
            {"element_id": "2.B.7.b", "skill_name": "Coordination", "importance": 66, "level": 4.62},
        ],
    },
    "21-1021.00": {
        "title": "Child, Family, and School Social Workers",
        "job_zone": 4,
        "sector": "Community/Social",
        "skills": [
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 84, "level": 5.88},
            {"element_id": "2.B.7.a", "skill_name": "Social Perceptiveness", "importance": 84, "level": 5.88},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 78, "level": 5.38},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 75, "level": 5.25},
            {"element_id": "2.B.3.a", "skill_name": "Writing", "importance": 72, "level": 5.0},
            {"element_id": "2.B.1.f", "skill_name": "Service Orientation", "importance": 81, "level": 5.62},
            {"element_id": "2.B.7.b", "skill_name": "Coordination", "importance": 69, "level": 4.88},
        ],
    },
    "21-1012.00": {
        "title": "Educational, Guidance, and Career Counselors",
        "job_zone": 4,
        "sector": "Community/Social",
        "skills": [
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 84, "level": 5.88},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 81, "level": 5.62},
            {"element_id": "2.B.7.a", "skill_name": "Social Perceptiveness", "importance": 81, "level": 5.62},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 78, "level": 5.38},
            {"element_id": "2.B.3.a", "skill_name": "Writing", "importance": 72, "level": 5.0},
            {"element_id": "2.B.1.f", "skill_name": "Service Orientation", "importance": 78, "level": 5.38},
            {"element_id": "2.B.3.d", "skill_name": "Instructing", "importance": 72, "level": 5.0},
        ],
    },
    "21-1093.00": {
        "title": "Social and Human Service Assistants",
        "job_zone": 2,
        "sector": "Community/Social",
        "skills": [
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 78, "level": 5.38},
            {"element_id": "2.B.7.a", "skill_name": "Social Perceptiveness", "importance": 75, "level": 5.25},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 72, "level": 5.0},
            {"element_id": "2.B.1.f", "skill_name": "Service Orientation", "importance": 78, "level": 5.38},
            {"element_id": "2.B.3.a", "skill_name": "Writing", "importance": 66, "level": 4.62},
            {"element_id": "2.B.7.b", "skill_name": "Coordination", "importance": 66, "level": 4.62},
        ],
    },
    "21-1014.00": {
        "title": "Mental Health Counselors",
        "job_zone": 4,
        "sector": "Community/Social",
        "skills": [
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 87, "level": 6.0},
            {"element_id": "2.B.7.a", "skill_name": "Social Perceptiveness", "importance": 84, "level": 5.88},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 78, "level": 5.38},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 78, "level": 5.38},
            {"element_id": "2.B.3.a", "skill_name": "Writing", "importance": 72, "level": 5.0},
            {"element_id": "2.B.1.f", "skill_name": "Service Orientation", "importance": 81, "level": 5.62},
        ],
    },
    "23-1011.00": {
        "title": "Lawyers",
        "job_zone": 5,
        "sector": "Legal",
        "skills": [
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 84, "level": 5.88},
            {"element_id": "2.B.3.a", "skill_name": "Writing", "importance": 87, "level": 6.0},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 84, "level": 5.88},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 87, "level": 6.0},
            {"element_id": "2.B.3.b", "skill_name": "Persuasion", "importance": 81, "level": 5.62},
            {"element_id": "2.B.3.c", "skill_name": "Negotiation", "importance": 81, "level": 5.62},
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 78, "level": 5.38},
        ],
    },
    "23-2011.00": {
        "title": "Paralegals and Legal Assistants",
        "job_zone": 3,
        "sector": "Legal",
        "skills": [
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 81, "level": 5.62},
            {"element_id": "2.B.3.a", "skill_name": "Writing", "importance": 78, "level": 5.38},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 75, "level": 5.25},
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 72, "level": 5.0},
            {"element_id": "2.B.6.b", "skill_name": "Time Management", "importance": 72, "level": 5.0},
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 75, "level": 5.25},
        ],
    },
    "23-1021.00": {
        "title": "Administrative Law Judges",
        "job_zone": 5,
        "sector": "Legal",
        "skills": [
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 84, "level": 5.88},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 87, "level": 6.0},
            {"element_id": "2.B.3.a", "skill_name": "Writing", "importance": 84, "level": 5.88},
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 81, "level": 5.62},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 78, "level": 5.38},
            {"element_id": "2.B.7.a", "skill_name": "Social Perceptiveness", "importance": 75, "level": 5.25},
            {"element_id": "2.B.8.e", "skill_name": "Systems Evaluation", "importance": 72, "level": 5.0},
        ],
    },
    "23-1023.00": {
        "title": "Judges and Magistrates",
        "job_zone": 5,
        "sector": "Legal",
        "skills": [
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 84, "level": 5.88},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 87, "level": 6.0},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 81, "level": 5.62},
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 84, "level": 5.88},
            {"element_id": "2.B.3.a", "skill_name": "Writing", "importance": 81, "level": 5.62},
            {"element_id": "2.B.7.a", "skill_name": "Social Perceptiveness", "importance": 78, "level": 5.38},
            {"element_id": "2.B.8.e", "skill_name": "Systems Evaluation", "importance": 75, "level": 5.25},
        ],
    },
    "25-2021.00": {
        "title": "Elementary School Teachers",
        "job_zone": 4,
        "sector": "Education",
        "skills": [
            {"element_id": "2.B.3.d", "skill_name": "Instructing", "importance": 84, "level": 5.88},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 81, "level": 5.62},
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 78, "level": 5.38},
            {"element_id": "2.B.7.a", "skill_name": "Social Perceptiveness", "importance": 78, "level": 5.38},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 72, "level": 5.0},
            {"element_id": "2.B.3.a", "skill_name": "Writing", "importance": 69, "level": 4.88},
            {"element_id": "2.B.6.b", "skill_name": "Time Management", "importance": 72, "level": 5.0},
            {"element_id": "2.B.6.a", "skill_name": "Learning Strategies", "importance": 75, "level": 5.25},
        ],
    },
    "25-2031.00": {
        "title": "Secondary School Teachers",
        "job_zone": 4,
        "sector": "Education",
        "skills": [
            {"element_id": "2.B.3.d", "skill_name": "Instructing", "importance": 84, "level": 5.88},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 78, "level": 5.38},
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 75, "level": 5.25},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 75, "level": 5.25},
            {"element_id": "2.B.3.a", "skill_name": "Writing", "importance": 72, "level": 5.0},
            {"element_id": "2.B.7.a", "skill_name": "Social Perceptiveness", "importance": 72, "level": 5.0},
            {"element_id": "2.B.6.b", "skill_name": "Time Management", "importance": 72, "level": 5.0},
            {"element_id": "2.B.6.a", "skill_name": "Learning Strategies", "importance": 72, "level": 5.0},
        ],
    },
    "25-1011.00": {
        "title": "Business Teachers, Postsecondary",
        "job_zone": 5,
        "sector": "Education",
        "skills": [
            {"element_id": "2.B.3.d", "skill_name": "Instructing", "importance": 84, "level": 5.88},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 81, "level": 5.62},
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 81, "level": 5.62},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 81, "level": 5.62},
            {"element_id": "2.B.3.a", "skill_name": "Writing", "importance": 78, "level": 5.38},
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 75, "level": 5.25},
            {"element_id": "2.B.6.a", "skill_name": "Learning Strategies", "importance": 75, "level": 5.25},
        ],
    },
    "25-9041.00": {
        "title": "Teacher Assistants",
        "job_zone": 2,
        "sector": "Education",
        "skills": [
            {"element_id": "2.B.3.d", "skill_name": "Instructing", "importance": 72, "level": 5.0},
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 75, "level": 5.25},
            {"element_id": "2.B.7.a", "skill_name": "Social Perceptiveness", "importance": 72, "level": 5.0},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 69, "level": 4.88},
            {"element_id": "2.B.1.f", "skill_name": "Service Orientation", "importance": 72, "level": 5.0},
            {"element_id": "2.B.6.b", "skill_name": "Time Management", "importance": 66, "level": 4.62},
        ],
    },
    "25-4013.00": {
        "title": "Museum Technicians and Conservators",
        "job_zone": 3,
        "sector": "Education",
        "skills": [
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 72, "level": 5.0},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 69, "level": 4.88},
            {"element_id": "2.B.6.c", "skill_name": "Quality Control Analysis", "importance": 75, "level": 5.25},
            {"element_id": "2.B.5.b", "skill_name": "Science", "importance": 66, "level": 4.62},
            {"element_id": "2.B.3.a", "skill_name": "Writing", "importance": 66, "level": 4.62},
            {"element_id": "2.B.6.b", "skill_name": "Time Management", "importance": 69, "level": 4.88},
        ],
    },
    "27-1024.00": {
        "title": "Graphic Designers",
        "job_zone": 3,
        "sector": "Arts/Entertainment",
        "skills": [
            {"element_id": "2.B.5.c", "skill_name": "Design", "importance": 84, "level": 5.88},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 72, "level": 5.0},
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 69, "level": 4.88},
            {"element_id": "2.B.6.b", "skill_name": "Time Management", "importance": 69, "level": 4.88},
            {"element_id": "2.B.7.b", "skill_name": "Coordination", "importance": 66, "level": 4.62},
            {"element_id": "2.B.8.b", "skill_name": "Complex Problem Solving", "importance": 66, "level": 4.62},
        ],
    },
    "27-2012.00": {
        "title": "Producers and Directors",
        "job_zone": 4,
        "sector": "Arts/Entertainment",
        "skills": [
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 78, "level": 5.38},
            {"element_id": "2.B.7.b", "skill_name": "Coordination", "importance": 81, "level": 5.62},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 75, "level": 5.25},
            {"element_id": "2.B.4.e", "skill_name": "Management of Personnel Resources", "importance": 78, "level": 5.38},
            {"element_id": "2.B.6.b", "skill_name": "Time Management", "importance": 78, "level": 5.38},
            {"element_id": "2.B.3.b", "skill_name": "Persuasion", "importance": 72, "level": 5.0},
            {"element_id": "2.B.3.c", "skill_name": "Negotiation", "importance": 72, "level": 5.0},
        ],
    },
    "27-3031.00": {
        "title": "Public Relations Specialists",
        "job_zone": 4,
        "sector": "Arts/Entertainment",
        "skills": [
            {"element_id": "2.B.3.a", "skill_name": "Writing", "importance": 84, "level": 5.88},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 81, "level": 5.62},
            {"element_id": "2.B.3.b", "skill_name": "Persuasion", "importance": 78, "level": 5.38},
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 75, "level": 5.25},
            {"element_id": "2.B.7.a", "skill_name": "Social Perceptiveness", "importance": 72, "level": 5.0},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 72, "level": 5.0},
            {"element_id": "2.B.6.b", "skill_name": "Time Management", "importance": 69, "level": 4.88},
        ],
    },
    "27-1014.00": {
        "title": "Special Effects Artists and Animators",
        "job_zone": 3,
        "sector": "Arts/Entertainment",
        "skills": [
            {"element_id": "2.B.5.c", "skill_name": "Design", "importance": 84, "level": 5.88},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 72, "level": 5.0},
            {"element_id": "2.B.8.b", "skill_name": "Complex Problem Solving", "importance": 75, "level": 5.25},
            {"element_id": "2.B.1.g", "skill_name": "Programming", "importance": 69, "level": 4.88},
            {"element_id": "2.B.6.b", "skill_name": "Time Management", "importance": 69, "level": 4.88},
        ],
    },
    "27-2022.00": {
        "title": "Coaches and Scouts",
        "job_zone": 3,
        "sector": "Arts/Entertainment",
        "skills": [
            {"element_id": "2.B.3.d", "skill_name": "Instructing", "importance": 81, "level": 5.62},
            {"element_id": "2.B.7.a", "skill_name": "Social Perceptiveness", "importance": 78, "level": 5.38},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 78, "level": 5.38},
            {"element_id": "2.B.7.b", "skill_name": "Coordination", "importance": 72, "level": 5.0},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 69, "level": 4.88},
            {"element_id": "2.B.7.c", "skill_name": "Monitoring", "importance": 72, "level": 5.0},
        ],
    },
    "29-1141.00": {
        "title": "Registered Nurses",
        "job_zone": 4,
        "sector": "Healthcare",
        "skills": [
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 81, "level": 5.62},
            {"element_id": "2.B.7.a", "skill_name": "Social Perceptiveness", "importance": 81, "level": 5.62},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 78, "level": 5.38},
            {"element_id": "2.B.1.f", "skill_name": "Service Orientation", "importance": 81, "level": 5.62},
            {"element_id": "2.B.7.b", "skill_name": "Coordination", "importance": 75, "level": 5.25},
            {"element_id": "2.B.7.c", "skill_name": "Monitoring", "importance": 78, "level": 5.38},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 72, "level": 5.0},
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 72, "level": 5.0},
        ],
    },
    "29-1071.00": {
        "title": "Physician Assistants",
        "job_zone": 5,
        "sector": "Healthcare",
        "skills": [
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 78, "level": 5.38},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 84, "level": 5.88},
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 81, "level": 5.62},
            {"element_id": "2.B.5.b", "skill_name": "Science", "importance": 78, "level": 5.38},
            {"element_id": "2.B.7.a", "skill_name": "Social Perceptiveness", "importance": 78, "level": 5.38},
            {"element_id": "2.B.1.f", "skill_name": "Service Orientation", "importance": 81, "level": 5.62},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 75, "level": 5.25},
            {"element_id": "2.B.8.b", "skill_name": "Complex Problem Solving", "importance": 72, "level": 5.0},
        ],
    },
    "29-2034.00": {
        "title": "Radiologic Technologists",
        "job_zone": 3,
        "sector": "Healthcare",
        "skills": [
            {"element_id": "2.B.7.c", "skill_name": "Monitoring", "importance": 78, "level": 5.38},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 72, "level": 5.0},
            {"element_id": "2.B.1.f", "skill_name": "Service Orientation", "importance": 75, "level": 5.25},
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 72, "level": 5.0},
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 69, "level": 4.88},
            {"element_id": "2.B.5.b", "skill_name": "Science", "importance": 66, "level": 4.62},
        ],
    },
    "29-1031.00": {
        "title": "Dietitians and Nutritionists",
        "job_zone": 4,
        "sector": "Healthcare",
        "skills": [
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 78, "level": 5.38},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 78, "level": 5.38},
            {"element_id": "2.B.5.b", "skill_name": "Science", "importance": 75, "level": 5.25},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 75, "level": 5.25},
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 75, "level": 5.25},
            {"element_id": "2.B.3.a", "skill_name": "Writing", "importance": 72, "level": 5.0},
            {"element_id": "2.B.1.f", "skill_name": "Service Orientation", "importance": 72, "level": 5.0},
        ],
    },
    "29-2061.00": {
        "title": "Licensed Practical Nurses",
        "job_zone": 3,
        "sector": "Healthcare",
        "skills": [
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 78, "level": 5.38},
            {"element_id": "2.B.1.f", "skill_name": "Service Orientation", "importance": 81, "level": 5.62},
            {"element_id": "2.B.7.a", "skill_name": "Social Perceptiveness", "importance": 75, "level": 5.25},
            {"element_id": "2.B.7.c", "skill_name": "Monitoring", "importance": 75, "level": 5.25},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 72, "level": 5.0},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 69, "level": 4.88},
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 69, "level": 4.88},
        ],
    },
    "31-1014.00": {
        "title": "Nursing Assistants",
        "job_zone": 2,
        "sector": "Healthcare Support",
        "skills": [
            {"element_id": "2.B.1.f", "skill_name": "Service Orientation", "importance": 81, "level": 5.62},
            {"element_id": "2.B.7.a", "skill_name": "Social Perceptiveness", "importance": 75, "level": 5.25},
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 75, "level": 5.25},
            {"element_id": "2.B.7.c", "skill_name": "Monitoring", "importance": 72, "level": 5.0},
            {"element_id": "2.B.7.b", "skill_name": "Coordination", "importance": 69, "level": 4.88},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 66, "level": 4.62},
        ],
    },
    "31-9091.00": {
        "title": "Dental Assistants",
        "job_zone": 3,
        "sector": "Healthcare Support",
        "skills": [
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 75, "level": 5.25},
            {"element_id": "2.B.1.f", "skill_name": "Service Orientation", "importance": 78, "level": 5.38},
            {"element_id": "2.B.7.c", "skill_name": "Monitoring", "importance": 72, "level": 5.0},
            {"element_id": "2.B.7.b", "skill_name": "Coordination", "importance": 72, "level": 5.0},
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 66, "level": 4.62},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 66, "level": 4.62},
        ],
    },
    "31-9092.00": {
        "title": "Medical Assistants",
        "job_zone": 3,
        "sector": "Healthcare Support",
        "skills": [
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 75, "level": 5.25},
            {"element_id": "2.B.1.f", "skill_name": "Service Orientation", "importance": 78, "level": 5.38},
            {"element_id": "2.B.7.a", "skill_name": "Social Perceptiveness", "importance": 72, "level": 5.0},
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 69, "level": 4.88},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 69, "level": 4.88},
            {"element_id": "2.B.6.b", "skill_name": "Time Management", "importance": 69, "level": 4.88},
        ],
    },
    "31-2021.00": {
        "title": "Physical Therapist Assistants",
        "job_zone": 3,
        "sector": "Healthcare Support",
        "skills": [
            {"element_id": "2.B.1.f", "skill_name": "Service Orientation", "importance": 81, "level": 5.62},
            {"element_id": "2.B.7.a", "skill_name": "Social Perceptiveness", "importance": 78, "level": 5.38},
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 78, "level": 5.38},
            {"element_id": "2.B.3.d", "skill_name": "Instructing", "importance": 75, "level": 5.25},
            {"element_id": "2.B.7.c", "skill_name": "Monitoring", "importance": 72, "level": 5.0},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 72, "level": 5.0},
        ],
    },
    "33-3051.00": {
        "title": "Police and Sheriff's Patrol Officers",
        "job_zone": 3,
        "sector": "Protective Service",
        "skills": [
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 78, "level": 5.38},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 75, "level": 5.25},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 75, "level": 5.25},
            {"element_id": "2.B.7.a", "skill_name": "Social Perceptiveness", "importance": 78, "level": 5.38},
            {"element_id": "2.B.7.c", "skill_name": "Monitoring", "importance": 72, "level": 5.0},
            {"element_id": "2.B.7.b", "skill_name": "Coordination", "importance": 69, "level": 4.88},
            {"element_id": "2.B.3.b", "skill_name": "Persuasion", "importance": 66, "level": 4.62},
        ],
    },
    "33-2011.00": {
        "title": "Firefighters",
        "job_zone": 3,
        "sector": "Protective Service",
        "skills": [
            {"element_id": "2.B.7.b", "skill_name": "Coordination", "importance": 81, "level": 5.62},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 72, "level": 5.0},
            {"element_id": "2.B.9.c", "skill_name": "Operations Monitoring", "importance": 78, "level": 5.38},
            {"element_id": "2.B.9.a", "skill_name": "Troubleshooting", "importance": 72, "level": 5.0},
            {"element_id": "2.B.4.h", "skill_name": "Equipment Maintenance", "importance": 69, "level": 4.88},
            {"element_id": "2.B.7.a", "skill_name": "Social Perceptiveness", "importance": 66, "level": 4.62},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 66, "level": 4.62},
        ],
    },
    "33-3012.00": {
        "title": "Correctional Officers and Jailers",
        "job_zone": 2,
        "sector": "Protective Service",
        "skills": [
            {"element_id": "2.B.7.a", "skill_name": "Social Perceptiveness", "importance": 78, "level": 5.38},
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 75, "level": 5.25},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 72, "level": 5.0},
            {"element_id": "2.B.7.c", "skill_name": "Monitoring", "importance": 72, "level": 5.0},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 69, "level": 4.88},
            {"element_id": "2.B.7.b", "skill_name": "Coordination", "importance": 66, "level": 4.62},
        ],
    },
    "33-9032.00": {
        "title": "Security Guards",
        "job_zone": 1,
        "sector": "Protective Service",
        "skills": [
            {"element_id": "2.B.7.c", "skill_name": "Monitoring", "importance": 72, "level": 5.0},
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 69, "level": 4.88},
            {"element_id": "2.B.7.a", "skill_name": "Social Perceptiveness", "importance": 69, "level": 4.88},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 66, "level": 4.62},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 63, "level": 4.38},
        ],
    },
    "35-2014.00": {
        "title": "Cooks, Restaurant",
        "job_zone": 2,
        "sector": "Food Service/Hospitality",
        "skills": [
            {"element_id": "2.B.7.c", "skill_name": "Monitoring", "importance": 72, "level": 5.0},
            {"element_id": "2.B.6.b", "skill_name": "Time Management", "importance": 72, "level": 5.0},
            {"element_id": "2.B.7.b", "skill_name": "Coordination", "importance": 69, "level": 4.88},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 63, "level": 4.38},
            {"element_id": "2.B.9.c", "skill_name": "Operations Monitoring", "importance": 69, "level": 4.88},
        ],
    },
    "35-1012.00": {
        "title": "First-Line Supervisors of Food Service",
        "job_zone": 3,
        "sector": "Food Service/Hospitality",
        "skills": [
            {"element_id": "2.B.7.b", "skill_name": "Coordination", "importance": 78, "level": 5.38},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 75, "level": 5.25},
            {"element_id": "2.B.6.b", "skill_name": "Time Management", "importance": 78, "level": 5.38},
            {"element_id": "2.B.4.e", "skill_name": "Management of Personnel Resources", "importance": 75, "level": 5.25},
            {"element_id": "2.B.7.c", "skill_name": "Monitoring", "importance": 72, "level": 5.0},
            {"element_id": "2.B.1.f", "skill_name": "Service Orientation", "importance": 72, "level": 5.0},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 69, "level": 4.88},
        ],
    },
    "35-3031.00": {
        "title": "Waiters and Waitresses",
        "job_zone": 1,
        "sector": "Food Service/Hospitality",
        "skills": [
            {"element_id": "2.B.1.f", "skill_name": "Service Orientation", "importance": 75, "level": 5.25},
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 72, "level": 5.0},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 69, "level": 4.88},
            {"element_id": "2.B.7.a", "skill_name": "Social Perceptiveness", "importance": 66, "level": 4.62},
            {"element_id": "2.B.6.b", "skill_name": "Time Management", "importance": 63, "level": 4.38},
        ],
    },
    "35-3023.00": {
        "title": "Fast Food and Counter Workers",
        "job_zone": 1,
        "sector": "Food Service/Hospitality",
        "skills": [
            {"element_id": "2.B.1.f", "skill_name": "Service Orientation", "importance": 72, "level": 5.0},
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 69, "level": 4.88},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 66, "level": 4.62},
            {"element_id": "2.B.6.b", "skill_name": "Time Management", "importance": 63, "level": 4.38},
            {"element_id": "2.B.7.b", "skill_name": "Coordination", "importance": 60, "level": 4.12},
        ],
    },
    "37-2011.00": {
        "title": "Janitors and Cleaners",
        "job_zone": 1,
        "sector": "Building/Grounds Maintenance",
        "skills": [
            {"element_id": "2.B.9.c", "skill_name": "Operations Monitoring", "importance": 66, "level": 4.62},
            {"element_id": "2.B.6.b", "skill_name": "Time Management", "importance": 63, "level": 4.38},
            {"element_id": "2.B.7.b", "skill_name": "Coordination", "importance": 60, "level": 4.12},
            {"element_id": "2.B.4.h", "skill_name": "Equipment Maintenance", "importance": 63, "level": 4.38},
            {"element_id": "2.B.7.c", "skill_name": "Monitoring", "importance": 60, "level": 4.12},
        ],
    },
    "37-1011.00": {
        "title": "First-Line Supervisors of Housekeeping",
        "job_zone": 3,
        "sector": "Building/Grounds Maintenance",
        "skills": [
            {"element_id": "2.B.7.b", "skill_name": "Coordination", "importance": 75, "level": 5.25},
            {"element_id": "2.B.4.e", "skill_name": "Management of Personnel Resources", "importance": 72, "level": 5.0},
            {"element_id": "2.B.6.b", "skill_name": "Time Management", "importance": 75, "level": 5.25},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 72, "level": 5.0},
            {"element_id": "2.B.7.c", "skill_name": "Monitoring", "importance": 69, "level": 4.88},
            {"element_id": "2.B.6.c", "skill_name": "Quality Control Analysis", "importance": 69, "level": 4.88},
        ],
    },
    "37-3011.00": {
        "title": "Landscaping and Groundskeeping Workers",
        "job_zone": 1,
        "sector": "Building/Grounds Maintenance",
        "skills": [
            {"element_id": "2.B.9.c", "skill_name": "Operations Monitoring", "importance": 66, "level": 4.62},
            {"element_id": "2.B.4.h", "skill_name": "Equipment Maintenance", "importance": 66, "level": 4.62},
            {"element_id": "2.B.7.b", "skill_name": "Coordination", "importance": 63, "level": 4.38},
            {"element_id": "2.B.6.b", "skill_name": "Time Management", "importance": 63, "level": 4.38},
            {"element_id": "2.B.9.b", "skill_name": "Equipment Selection", "importance": 60, "level": 4.12},
        ],
    },
    "37-2021.00": {
        "title": "Pest Control Workers",
        "job_zone": 2,
        "sector": "Building/Grounds Maintenance",
        "skills": [
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 66, "level": 4.62},
            {"element_id": "2.B.1.f", "skill_name": "Service Orientation", "importance": 69, "level": 4.88},
            {"element_id": "2.B.9.c", "skill_name": "Operations Monitoring", "importance": 69, "level": 4.88},
            {"element_id": "2.B.5.b", "skill_name": "Science", "importance": 63, "level": 4.38},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 66, "level": 4.62},
            {"element_id": "2.B.9.a", "skill_name": "Troubleshooting", "importance": 66, "level": 4.62},
        ],
    },
    "39-5012.00": {
        "title": "Hairdressers, Hairstylists, and Cosmetologists",
        "job_zone": 2,
        "sector": "Personal Care",
        "skills": [
            {"element_id": "2.B.1.f", "skill_name": "Service Orientation", "importance": 78, "level": 5.38},
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 75, "level": 5.25},
            {"element_id": "2.B.7.a", "skill_name": "Social Perceptiveness", "importance": 72, "level": 5.0},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 72, "level": 5.0},
            {"element_id": "2.B.6.b", "skill_name": "Time Management", "importance": 69, "level": 4.88},
            {"element_id": "2.B.5.c", "skill_name": "Design", "importance": 66, "level": 4.62},
        ],
    },
    "39-9011.00": {
        "title": "Childcare Workers",
        "job_zone": 1,
        "sector": "Personal Care",
        "skills": [
            {"element_id": "2.B.1.f", "skill_name": "Service Orientation", "importance": 78, "level": 5.38},
            {"element_id": "2.B.7.a", "skill_name": "Social Perceptiveness", "importance": 75, "level": 5.25},
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 75, "level": 5.25},
            {"element_id": "2.B.3.d", "skill_name": "Instructing", "importance": 72, "level": 5.0},
            {"element_id": "2.B.7.c", "skill_name": "Monitoring", "importance": 69, "level": 4.88},
        ],
    },
    "39-1014.00": {
        "title": "First-Line Supervisors of Entertainment Workers",
        "job_zone": 3,
        "sector": "Personal Care",
        "skills": [
            {"element_id": "2.B.7.b", "skill_name": "Coordination", "importance": 75, "level": 5.25},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 75, "level": 5.25},
            {"element_id": "2.B.4.e", "skill_name": "Management of Personnel Resources", "importance": 72, "level": 5.0},
            {"element_id": "2.B.6.b", "skill_name": "Time Management", "importance": 75, "level": 5.25},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 69, "level": 4.88},
            {"element_id": "2.B.7.a", "skill_name": "Social Perceptiveness", "importance": 69, "level": 4.88},
        ],
    },
    "39-9031.00": {
        "title": "Exercise Trainers and Group Fitness Instructors",
        "job_zone": 3,
        "sector": "Personal Care",
        "skills": [
            {"element_id": "2.B.3.d", "skill_name": "Instructing", "importance": 81, "level": 5.62},
            {"element_id": "2.B.1.f", "skill_name": "Service Orientation", "importance": 78, "level": 5.38},
            {"element_id": "2.B.7.a", "skill_name": "Social Perceptiveness", "importance": 75, "level": 5.25},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 75, "level": 5.25},
            {"element_id": "2.B.7.c", "skill_name": "Monitoring", "importance": 72, "level": 5.0},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 66, "level": 4.62},
        ],
    },
    "39-4031.00": {
        "title": "Morticians and Funeral Attendants",
        "job_zone": 2,
        "sector": "Personal Care",
        "skills": [
            {"element_id": "2.B.1.f", "skill_name": "Service Orientation", "importance": 78, "level": 5.38},
            {"element_id": "2.B.7.a", "skill_name": "Social Perceptiveness", "importance": 75, "level": 5.25},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 72, "level": 5.0},
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 72, "level": 5.0},
            {"element_id": "2.B.7.b", "skill_name": "Coordination", "importance": 69, "level": 4.88},
        ],
    },
    "41-2031.00": {
        "title": "Retail Salespersons",
        "job_zone": 2,
        "sector": "Retail/Sales",
        "skills": [
            {"element_id": "2.B.1.f", "skill_name": "Service Orientation", "importance": 78, "level": 5.38},
            {"element_id": "2.B.3.b", "skill_name": "Persuasion", "importance": 72, "level": 5.0},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 72, "level": 5.0},
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 72, "level": 5.0},
            {"element_id": "2.B.7.a", "skill_name": "Social Perceptiveness", "importance": 69, "level": 4.88},
            {"element_id": "2.B.3.c", "skill_name": "Negotiation", "importance": 63, "level": 4.38},
        ],
    },
    "41-3031.00": {
        "title": "Securities and Financial Services Sales Agents",
        "job_zone": 4,
        "sector": "Retail/Sales",
        "skills": [
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 81, "level": 5.62},
            {"element_id": "2.B.3.b", "skill_name": "Persuasion", "importance": 84, "level": 5.88},
            {"element_id": "2.B.3.c", "skill_name": "Negotiation", "importance": 78, "level": 5.38},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 78, "level": 5.38},
            {"element_id": "2.B.5.a", "skill_name": "Mathematics", "importance": 75, "level": 5.25},
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 75, "level": 5.25},
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 72, "level": 5.0},
        ],
    },
    "41-1011.00": {
        "title": "First-Line Supervisors of Retail Sales",
        "job_zone": 3,
        "sector": "Retail/Sales",
        "skills": [
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 78, "level": 5.38},
            {"element_id": "2.B.4.e", "skill_name": "Management of Personnel Resources", "importance": 78, "level": 5.38},
            {"element_id": "2.B.7.b", "skill_name": "Coordination", "importance": 75, "level": 5.25},
            {"element_id": "2.B.3.b", "skill_name": "Persuasion", "importance": 72, "level": 5.0},
            {"element_id": "2.B.6.b", "skill_name": "Time Management", "importance": 75, "level": 5.25},
            {"element_id": "2.B.1.f", "skill_name": "Service Orientation", "importance": 72, "level": 5.0},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 69, "level": 4.88},
        ],
    },
    "41-4012.00": {
        "title": "Sales Representatives, Wholesale",
        "job_zone": 3,
        "sector": "Retail/Sales",
        "skills": [
            {"element_id": "2.B.3.b", "skill_name": "Persuasion", "importance": 81, "level": 5.62},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 78, "level": 5.38},
            {"element_id": "2.B.3.c", "skill_name": "Negotiation", "importance": 75, "level": 5.25},
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 72, "level": 5.0},
            {"element_id": "2.B.1.f", "skill_name": "Service Orientation", "importance": 72, "level": 5.0},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 69, "level": 4.88},
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 66, "level": 4.62},
        ],
    },
    "41-9022.00": {
        "title": "Real Estate Sales Agents",
        "job_zone": 3,
        "sector": "Retail/Sales",
        "skills": [
            {"element_id": "2.B.3.b", "skill_name": "Persuasion", "importance": 81, "level": 5.62},
            {"element_id": "2.B.3.c", "skill_name": "Negotiation", "importance": 81, "level": 5.62},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 78, "level": 5.38},
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 75, "level": 5.25},
            {"element_id": "2.B.7.a", "skill_name": "Social Perceptiveness", "importance": 72, "level": 5.0},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 69, "level": 4.88},
            {"element_id": "2.B.3.a", "skill_name": "Writing", "importance": 66, "level": 4.62},
        ],
    },
    "43-4051.00": {
        "title": "Customer Service Representatives",
        "job_zone": 2,
        "sector": "Office/Admin Support",
        "skills": [
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 78, "level": 5.38},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 75, "level": 5.25},
            {"element_id": "2.B.1.f", "skill_name": "Service Orientation", "importance": 78, "level": 5.38},
            {"element_id": "2.B.7.a", "skill_name": "Social Perceptiveness", "importance": 72, "level": 5.0},
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 69, "level": 4.88},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 66, "level": 4.62},
        ],
    },
    "43-6014.00": {
        "title": "Secretaries and Administrative Assistants",
        "job_zone": 3,
        "sector": "Office/Admin Support",
        "skills": [
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 72, "level": 5.0},
            {"element_id": "2.B.3.a", "skill_name": "Writing", "importance": 75, "level": 5.25},
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 72, "level": 5.0},
            {"element_id": "2.B.6.b", "skill_name": "Time Management", "importance": 75, "level": 5.25},
            {"element_id": "2.B.7.b", "skill_name": "Coordination", "importance": 72, "level": 5.0},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 69, "level": 4.88},
        ],
    },
    "43-3031.00": {
        "title": "Bookkeeping, Accounting, and Auditing Clerks",
        "job_zone": 3,
        "sector": "Office/Admin Support",
        "skills": [
            {"element_id": "2.B.5.a", "skill_name": "Mathematics", "importance": 75, "level": 5.25},
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 72, "level": 5.0},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 69, "level": 4.88},
            {"element_id": "2.B.6.b", "skill_name": "Time Management", "importance": 72, "level": 5.0},
            {"element_id": "2.B.7.c", "skill_name": "Monitoring", "importance": 69, "level": 4.88},
        ],
    },
    "43-9061.00": {
        "title": "Office Clerks, General",
        "job_zone": 2,
        "sector": "Office/Admin Support",
        "skills": [
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 69, "level": 4.88},
            {"element_id": "2.B.3.a", "skill_name": "Writing", "importance": 66, "level": 4.62},
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 66, "level": 4.62},
            {"element_id": "2.B.6.b", "skill_name": "Time Management", "importance": 69, "level": 4.88},
            {"element_id": "2.B.7.b", "skill_name": "Coordination", "importance": 63, "level": 4.38},
        ],
    },
    "43-1011.00": {
        "title": "First-Line Supervisors of Office Workers",
        "job_zone": 3,
        "sector": "Office/Admin Support",
        "skills": [
            {"element_id": "2.B.7.b", "skill_name": "Coordination", "importance": 78, "level": 5.38},
            {"element_id": "2.B.4.e", "skill_name": "Management of Personnel Resources", "importance": 78, "level": 5.38},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 75, "level": 5.25},
            {"element_id": "2.B.6.b", "skill_name": "Time Management", "importance": 78, "level": 5.38},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 72, "level": 5.0},
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 72, "level": 5.0},
            {"element_id": "2.B.3.a", "skill_name": "Writing", "importance": 69, "level": 4.88},
        ],
    },
    "45-2092.00": {
        "title": "Farmworkers and Laborers",
        "job_zone": 1,
        "sector": "Farming/Agriculture",
        "skills": [
            {"element_id": "2.B.9.c", "skill_name": "Operations Monitoring", "importance": 66, "level": 4.62},
            {"element_id": "2.B.7.b", "skill_name": "Coordination", "importance": 63, "level": 4.38},
            {"element_id": "2.B.4.h", "skill_name": "Equipment Maintenance", "importance": 63, "level": 4.38},
            {"element_id": "2.B.7.c", "skill_name": "Monitoring", "importance": 60, "level": 4.12},
            {"element_id": "2.B.9.b", "skill_name": "Equipment Selection", "importance": 60, "level": 4.12},
        ],
    },
    "45-2011.00": {
        "title": "Agricultural Inspectors",
        "job_zone": 3,
        "sector": "Farming/Agriculture",
        "skills": [
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 72, "level": 5.0},
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 72, "level": 5.0},
            {"element_id": "2.B.7.c", "skill_name": "Monitoring", "importance": 75, "level": 5.25},
            {"element_id": "2.B.6.c", "skill_name": "Quality Control Analysis", "importance": 75, "level": 5.25},
            {"element_id": "2.B.3.a", "skill_name": "Writing", "importance": 69, "level": 4.88},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 66, "level": 4.62},
        ],
    },
    "45-1011.00": {
        "title": "First-Line Supervisors of Farming Workers",
        "job_zone": 3,
        "sector": "Farming/Agriculture",
        "skills": [
            {"element_id": "2.B.7.b", "skill_name": "Coordination", "importance": 75, "level": 5.25},
            {"element_id": "2.B.4.e", "skill_name": "Management of Personnel Resources", "importance": 72, "level": 5.0},
            {"element_id": "2.B.6.b", "skill_name": "Time Management", "importance": 75, "level": 5.25},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 72, "level": 5.0},
            {"element_id": "2.B.4.g", "skill_name": "Management of Material Resources", "importance": 72, "level": 5.0},
            {"element_id": "2.B.7.c", "skill_name": "Monitoring", "importance": 69, "level": 4.88},
        ],
    },
    "45-4011.00": {
        "title": "Forest and Conservation Workers",
        "job_zone": 2,
        "sector": "Farming/Agriculture",
        "skills": [
            {"element_id": "2.B.9.c", "skill_name": "Operations Monitoring", "importance": 72, "level": 5.0},
            {"element_id": "2.B.4.h", "skill_name": "Equipment Maintenance", "importance": 69, "level": 4.88},
            {"element_id": "2.B.7.b", "skill_name": "Coordination", "importance": 66, "level": 4.62},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 66, "level": 4.62},
            {"element_id": "2.B.9.b", "skill_name": "Equipment Selection", "importance": 66, "level": 4.62},
        ],
    },
    "47-2111.00": {
        "title": "Electricians",
        "job_zone": 3,
        "sector": "Construction",
        "skills": [
            {"element_id": "2.B.9.a", "skill_name": "Troubleshooting", "importance": 81, "level": 5.62},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 75, "level": 5.25},
            {"element_id": "2.B.9.c", "skill_name": "Operations Monitoring", "importance": 78, "level": 5.38},
            {"element_id": "2.B.5.a", "skill_name": "Mathematics", "importance": 69, "level": 4.88},
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 69, "level": 4.88},
            {"element_id": "2.B.4.h", "skill_name": "Equipment Maintenance", "importance": 75, "level": 5.25},
            {"element_id": "2.B.9.b", "skill_name": "Equipment Selection", "importance": 72, "level": 5.0},
        ],
    },
    "47-2031.00": {
        "title": "Carpenters",
        "job_zone": 2,
        "sector": "Construction",
        "skills": [
            {"element_id": "2.B.5.a", "skill_name": "Mathematics", "importance": 69, "level": 4.88},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 66, "level": 4.62},
            {"element_id": "2.B.9.c", "skill_name": "Operations Monitoring", "importance": 72, "level": 5.0},
            {"element_id": "2.B.4.h", "skill_name": "Equipment Maintenance", "importance": 72, "level": 5.0},
            {"element_id": "2.B.9.b", "skill_name": "Equipment Selection", "importance": 72, "level": 5.0},
            {"element_id": "2.B.5.c", "skill_name": "Design", "importance": 66, "level": 4.62},
        ],
    },
    "47-2152.00": {
        "title": "Plumbers",
        "job_zone": 3,
        "sector": "Construction",
        "skills": [
            {"element_id": "2.B.9.a", "skill_name": "Troubleshooting", "importance": 78, "level": 5.38},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 72, "level": 5.0},
            {"element_id": "2.B.9.c", "skill_name": "Operations Monitoring", "importance": 75, "level": 5.25},
            {"element_id": "2.B.4.h", "skill_name": "Equipment Maintenance", "importance": 75, "level": 5.25},
            {"element_id": "2.B.5.a", "skill_name": "Mathematics", "importance": 66, "level": 4.62},
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 66, "level": 4.62},
        ],
    },
    "47-1011.00": {
        "title": "First-Line Supervisors of Construction Trades",
        "job_zone": 3,
        "sector": "Construction",
        "skills": [
            {"element_id": "2.B.7.b", "skill_name": "Coordination", "importance": 81, "level": 5.62},
            {"element_id": "2.B.4.e", "skill_name": "Management of Personnel Resources", "importance": 78, "level": 5.38},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 75, "level": 5.25},
            {"element_id": "2.B.6.b", "skill_name": "Time Management", "importance": 78, "level": 5.38},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 75, "level": 5.25},
            {"element_id": "2.B.6.c", "skill_name": "Quality Control Analysis", "importance": 72, "level": 5.0},
            {"element_id": "2.B.4.g", "skill_name": "Management of Material Resources", "importance": 72, "level": 5.0},
        ],
    },
    "47-2061.00": {
        "title": "Construction Laborers",
        "job_zone": 1,
        "sector": "Construction",
        "skills": [
            {"element_id": "2.B.9.c", "skill_name": "Operations Monitoring", "importance": 69, "level": 4.88},
            {"element_id": "2.B.7.b", "skill_name": "Coordination", "importance": 66, "level": 4.62},
            {"element_id": "2.B.4.h", "skill_name": "Equipment Maintenance", "importance": 66, "level": 4.62},
            {"element_id": "2.B.9.b", "skill_name": "Equipment Selection", "importance": 63, "level": 4.38},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 60, "level": 4.12},
        ],
    },
    "49-3023.00": {
        "title": "Automotive Service Technicians",
        "job_zone": 3,
        "sector": "Installation/Maintenance",
        "skills": [
            {"element_id": "2.B.9.a", "skill_name": "Troubleshooting", "importance": 84, "level": 5.75},
            {"element_id": "2.B.4.h", "skill_name": "Equipment Maintenance", "importance": 81, "level": 5.62},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 72, "level": 5.0},
            {"element_id": "2.B.8.b", "skill_name": "Complex Problem Solving", "importance": 72, "level": 5.0},
            {"element_id": "2.B.9.c", "skill_name": "Operations Monitoring", "importance": 75, "level": 5.25},
            {"element_id": "2.B.9.b", "skill_name": "Equipment Selection", "importance": 75, "level": 5.25},
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 66, "level": 4.62},
        ],
    },
    "49-9021.00": {
        "title": "Heating, AC, and Refrigeration Mechanics",
        "job_zone": 3,
        "sector": "Installation/Maintenance",
        "skills": [
            {"element_id": "2.B.9.a", "skill_name": "Troubleshooting", "importance": 81, "level": 5.62},
            {"element_id": "2.B.4.h", "skill_name": "Equipment Maintenance", "importance": 81, "level": 5.62},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 72, "level": 5.0},
            {"element_id": "2.B.9.c", "skill_name": "Operations Monitoring", "importance": 75, "level": 5.25},
            {"element_id": "2.B.9.b", "skill_name": "Equipment Selection", "importance": 75, "level": 5.25},
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 66, "level": 4.62},
        ],
    },
    "49-1011.00": {
        "title": "First-Line Supervisors of Mechanics",
        "job_zone": 3,
        "sector": "Installation/Maintenance",
        "skills": [
            {"element_id": "2.B.7.b", "skill_name": "Coordination", "importance": 78, "level": 5.38},
            {"element_id": "2.B.4.e", "skill_name": "Management of Personnel Resources", "importance": 75, "level": 5.25},
            {"element_id": "2.B.9.a", "skill_name": "Troubleshooting", "importance": 78, "level": 5.38},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 75, "level": 5.25},
            {"element_id": "2.B.6.b", "skill_name": "Time Management", "importance": 75, "level": 5.25},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 72, "level": 5.0},
            {"element_id": "2.B.6.c", "skill_name": "Quality Control Analysis", "importance": 72, "level": 5.0},
        ],
    },
    "49-9041.00": {
        "title": "Industrial Machinery Mechanics",
        "job_zone": 3,
        "sector": "Installation/Maintenance",
        "skills": [
            {"element_id": "2.B.9.a", "skill_name": "Troubleshooting", "importance": 84, "level": 5.88},
            {"element_id": "2.B.4.h", "skill_name": "Equipment Maintenance", "importance": 84, "level": 5.88},
            {"element_id": "2.B.9.c", "skill_name": "Operations Monitoring", "importance": 78, "level": 5.38},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 72, "level": 5.0},
            {"element_id": "2.B.9.b", "skill_name": "Equipment Selection", "importance": 78, "level": 5.38},
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 66, "level": 4.62},
        ],
    },
    "49-2022.00": {
        "title": "Telecommunications Equipment Installers",
        "job_zone": 3,
        "sector": "Installation/Maintenance",
        "skills": [
            {"element_id": "2.B.9.a", "skill_name": "Troubleshooting", "importance": 78, "level": 5.38},
            {"element_id": "2.B.4.h", "skill_name": "Equipment Maintenance", "importance": 75, "level": 5.25},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 72, "level": 5.0},
            {"element_id": "2.B.9.c", "skill_name": "Operations Monitoring", "importance": 72, "level": 5.0},
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 69, "level": 4.88},
            {"element_id": "2.B.9.b", "skill_name": "Equipment Selection", "importance": 69, "level": 4.88},
            {"element_id": "2.B.8.b", "skill_name": "Complex Problem Solving", "importance": 66, "level": 4.62},
        ],
    },
    "51-4121.00": {
        "title": "Welders, Cutters, Solderers, and Brazers",
        "job_zone": 2,
        "sector": "Production/Manufacturing",
        "skills": [
            {"element_id": "2.B.9.c", "skill_name": "Operations Monitoring", "importance": 78, "level": 5.38},
            {"element_id": "2.B.6.c", "skill_name": "Quality Control Analysis", "importance": 75, "level": 5.25},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 66, "level": 4.62},
            {"element_id": "2.B.4.h", "skill_name": "Equipment Maintenance", "importance": 72, "level": 5.0},
            {"element_id": "2.B.9.b", "skill_name": "Equipment Selection", "importance": 72, "level": 5.0},
            {"element_id": "2.B.5.a", "skill_name": "Mathematics", "importance": 63, "level": 4.38},
        ],
    },
    "51-1011.00": {
        "title": "First-Line Supervisors of Production Workers",
        "job_zone": 3,
        "sector": "Production/Manufacturing",
        "skills": [
            {"element_id": "2.B.7.b", "skill_name": "Coordination", "importance": 78, "level": 5.38},
            {"element_id": "2.B.4.e", "skill_name": "Management of Personnel Resources", "importance": 78, "level": 5.38},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 75, "level": 5.25},
            {"element_id": "2.B.6.b", "skill_name": "Time Management", "importance": 78, "level": 5.38},
            {"element_id": "2.B.6.c", "skill_name": "Quality Control Analysis", "importance": 75, "level": 5.25},
            {"element_id": "2.B.7.c", "skill_name": "Monitoring", "importance": 72, "level": 5.0},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 72, "level": 5.0},
        ],
    },
    "51-9111.00": {
        "title": "Packaging and Filling Machine Operators",
        "job_zone": 1,
        "sector": "Production/Manufacturing",
        "skills": [
            {"element_id": "2.B.9.c", "skill_name": "Operations Monitoring", "importance": 72, "level": 5.0},
            {"element_id": "2.B.6.c", "skill_name": "Quality Control Analysis", "importance": 69, "level": 4.88},
            {"element_id": "2.B.7.c", "skill_name": "Monitoring", "importance": 66, "level": 4.62},
            {"element_id": "2.B.7.b", "skill_name": "Coordination", "importance": 63, "level": 4.38},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 60, "level": 4.12},
        ],
    },
    "51-4041.00": {
        "title": "Machinists",
        "job_zone": 3,
        "sector": "Production/Manufacturing",
        "skills": [
            {"element_id": "2.B.9.c", "skill_name": "Operations Monitoring", "importance": 81, "level": 5.62},
            {"element_id": "2.B.6.c", "skill_name": "Quality Control Analysis", "importance": 78, "level": 5.38},
            {"element_id": "2.B.5.a", "skill_name": "Mathematics", "importance": 75, "level": 5.25},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 72, "level": 5.0},
            {"element_id": "2.B.9.b", "skill_name": "Equipment Selection", "importance": 75, "level": 5.25},
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 66, "level": 4.62},
        ],
    },
    "51-2092.00": {
        "title": "Team Assemblers",
        "job_zone": 1,
        "sector": "Production/Manufacturing",
        "skills": [
            {"element_id": "2.B.9.c", "skill_name": "Operations Monitoring", "importance": 72, "level": 5.0},
            {"element_id": "2.B.6.c", "skill_name": "Quality Control Analysis", "importance": 69, "level": 4.88},
            {"element_id": "2.B.7.b", "skill_name": "Coordination", "importance": 66, "level": 4.62},
            {"element_id": "2.B.7.c", "skill_name": "Monitoring", "importance": 63, "level": 4.38},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 60, "level": 4.12},
        ],
    },
    "53-3032.00": {
        "title": "Heavy and Tractor-Trailer Truck Drivers",
        "job_zone": 2,
        "sector": "Transportation",
        "skills": [
            {"element_id": "2.B.9.c", "skill_name": "Operations Monitoring", "importance": 75, "level": 5.25},
            {"element_id": "2.B.7.c", "skill_name": "Monitoring", "importance": 72, "level": 5.0},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 66, "level": 4.62},
            {"element_id": "2.B.6.b", "skill_name": "Time Management", "importance": 69, "level": 4.88},
            {"element_id": "2.B.9.a", "skill_name": "Troubleshooting", "importance": 66, "level": 4.62},
        ],
    },
    "53-2011.00": {
        "title": "Airline Pilots",
        "job_zone": 5,
        "sector": "Transportation",
        "skills": [
            {"element_id": "2.B.9.c", "skill_name": "Operations Monitoring", "importance": 84, "level": 5.88},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 84, "level": 5.88},
            {"element_id": "2.B.8.b", "skill_name": "Complex Problem Solving", "importance": 81, "level": 5.62},
            {"element_id": "2.B.7.c", "skill_name": "Monitoring", "importance": 81, "level": 5.62},
            {"element_id": "2.B.7.b", "skill_name": "Coordination", "importance": 78, "level": 5.38},
            {"element_id": "2.B.5.a", "skill_name": "Mathematics", "importance": 72, "level": 5.0},
            {"element_id": "2.B.8.d", "skill_name": "Systems Analysis", "importance": 72, "level": 5.0},
        ],
    },
    "53-1048.00": {
        "title": "Supervisors of Transportation Workers",
        "job_zone": 3,
        "sector": "Transportation",
        "skills": [
            {"element_id": "2.B.7.b", "skill_name": "Coordination", "importance": 78, "level": 5.38},
            {"element_id": "2.B.4.e", "skill_name": "Management of Personnel Resources", "importance": 75, "level": 5.25},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 72, "level": 5.0},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 75, "level": 5.25},
            {"element_id": "2.B.6.b", "skill_name": "Time Management", "importance": 78, "level": 5.38},
            {"element_id": "2.B.7.c", "skill_name": "Monitoring", "importance": 72, "level": 5.0},
        ],
    },
    "53-6051.00": {
        "title": "Transportation Inspectors",
        "job_zone": 3,
        "sector": "Transportation",
        "skills": [
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 75, "level": 5.25},
            {"element_id": "2.B.7.c", "skill_name": "Monitoring", "importance": 78, "level": 5.38},
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 72, "level": 5.0},
            {"element_id": "2.B.6.c", "skill_name": "Quality Control Analysis", "importance": 75, "level": 5.25},
            {"element_id": "2.B.3.a", "skill_name": "Writing", "importance": 69, "level": 4.88},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 66, "level": 4.62},
        ],
    },
    "53-4011.00": {
        "title": "Locomotive Engineers",
        "job_zone": 3,
        "sector": "Transportation",
        "skills": [
            {"element_id": "2.B.9.c", "skill_name": "Operations Monitoring", "importance": 81, "level": 5.62},
            {"element_id": "2.B.7.c", "skill_name": "Monitoring", "importance": 78, "level": 5.38},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 72, "level": 5.0},
            {"element_id": "2.B.9.a", "skill_name": "Troubleshooting", "importance": 72, "level": 5.0},
            {"element_id": "2.B.7.b", "skill_name": "Coordination", "importance": 69, "level": 4.88},
            {"element_id": "2.B.6.b", "skill_name": "Time Management", "importance": 69, "level": 4.88},
        ],
    },
}


# ============================================================================
# 6 diverse user personas
# ============================================================================

PERSONA_TECH_WORKER = {
    "name": "Alex - Senior Software Engineer",
    "sector": "Information/Tech",
    "skill_ratings": {
        "2.B.1.a": 3,
        "2.B.8.a": 4,
        "2.B.8.b": 4,
        "2.B.8.d": 3,
        "2.B.1.g": 4,
        "2.B.3.a": 3,
        "2.B.5.a": 3,
        "2.B.6.b": 3,
        "2.B.8.e": 3,
        "2.B.9.a": 3,
    },
}

PERSONA_HEALTHCARE_WORKER = {
    "name": "Maria - Registered Nurse",
    "sector": "Healthcare",
    "skill_ratings": {
        "2.B.1.a": 3,
        "2.B.2.a": 4,
        "2.B.4.a": 4,
        "2.B.7.a": 4,
        "2.B.1.f": 4,
        "2.B.8.a": 3,
        "2.B.7.b": 3,
        "2.B.7.c": 3,
        "2.B.3.a": 2,
        "2.B.1.g": 0,
    },
}

PERSONA_BLUE_COLLAR = {
    "name": "Robert - Auto Mechanic",
    "sector": "Installation/Maintenance",
    "skill_ratings": {
        "2.B.9.a": 4,
        "2.B.4.h": 4,
        "2.B.8.b": 3,
        "2.B.8.a": 3,
        "2.B.9.b": 3,
        "2.B.9.c": 3,
        "2.B.6.c": 2,
        "2.B.1.a": 2,
        "2.B.4.a": 2,
        "2.B.1.g": 0,
    },
}

PERSONA_BUSINESS_PRO = {
    "name": "Sarah - Management Consultant",
    "sector": "Financial",
    "skill_ratings": {
        "2.B.1.a": 4,
        "2.B.3.a": 4,
        "2.B.4.a": 4,
        "2.B.8.a": 4,
        "2.B.8.b": 3,
        "2.B.3.b": 3,
        "2.B.3.c": 3,
        "2.B.4.e": 3,
        "2.B.4.f": 3,
        "2.B.6.b": 3,
        "2.B.7.b": 3,
    },
}

PERSONA_CREATIVE = {
    "name": "Jordan - Graphic Designer",
    "sector": "Arts/Entertainment",
    "skill_ratings": {
        "2.B.5.c": 4,
        "2.B.8.a": 3,
        "2.B.8.b": 3,
        "2.B.1.a": 3,
        "2.B.6.b": 3,
        "2.B.7.b": 2,
        "2.B.3.a": 3,
        "2.B.1.g": 2,
        "2.B.4.a": 2,
        "2.B.2.a": 2,
    },
}

PERSONA_SERVICE = {
    "name": "Diego - Restaurant Manager",
    "sector": "Food Service/Hospitality",
    "skill_ratings": {
        "2.B.4.a": 4,
        "2.B.2.a": 4,
        "2.B.7.a": 3,
        "2.B.1.f": 4,
        "2.B.7.b": 3,
        "2.B.4.e": 3,
        "2.B.6.b": 3,
        "2.B.3.b": 2,
        "2.B.1.a": 2,
        "2.B.8.a": 2,
    },
}

ALL_PERSONAS = [
    PERSONA_TECH_WORKER,
    PERSONA_HEALTHCARE_WORKER,
    PERSONA_BLUE_COLLAR,
    PERSONA_BUSINESS_PRO,
    PERSONA_CREATIVE,
    PERSONA_SERVICE,
]


# SOC prefix to sector name mapping
SOC_SECTORS = {
    "11": "Management",
    "13": "Financial",
    "15": "Information/Tech",
    "17": "Engineering/Construction",
    "19": "Science/Professional",
    "21": "Community/Social",
    "23": "Legal",
    "25": "Education",
    "27": "Arts/Entertainment",
    "29": "Healthcare",
    "31": "Healthcare Support",
    "33": "Protective Service",
    "35": "Food Service/Hospitality",
    "37": "Building/Grounds Maintenance",
    "39": "Personal Care",
    "41": "Retail/Sales",
    "43": "Office/Admin Support",
    "45": "Farming/Agriculture",
    "47": "Construction",
    "49": "Installation/Maintenance",
    "51": "Production/Manufacturing",
    "53": "Transportation",
}


def _score_profession(scorer, onet_code, persona):
    """Score a profession against a persona."""
    prof = PROFESSIONS_100[onet_code]
    return scorer.score_occupation(
        onet_code=onet_code,
        occupation_title=prof["title"],
        occupation_skills=prof["skills"],
        user_skill_ratings=persona["skill_ratings"],
        current_job_zone=3,
        target_job_zone=prof["job_zone"],
    )


# ============================================================================
# Test Classes
# ============================================================================

class TestAllProfessionsScoring:
    """Test scoring validity across all 100 professions."""

    def setup_method(self):
        self.scorer = BaselineScorer()

    def test_all_100_professions_produce_valid_scores(self):
        """Score all 100 professions for each persona and validate output."""
        for persona in ALL_PERSONAS:
            for onet_code in PROFESSIONS_100:
                score = _score_profession(self.scorer, onet_code, persona)
                assert isinstance(score, OccupationScore), (
                    f"Invalid score type for {onet_code} with {persona['name']}"
                )
                assert score.onet_code == onet_code

    def test_score_ranges_valid(self):
        """Verify match_score and gap_severity are in [0, 100]."""
        for persona in ALL_PERSONAS:
            for onet_code in PROFESSIONS_100:
                score = _score_profession(self.scorer, onet_code, persona)
                assert 0 <= score.match_score <= 100, (
                    f"match_score {score.match_score} out of range for {onet_code}"
                )
                assert 0 <= score.gap_severity <= 100, (
                    f"gap_severity {score.gap_severity} out of range for {onet_code}"
                )

    def test_bucket_always_valid(self):
        """All buckets must be one of the three valid values."""
        valid_buckets = {"ready_now", "trainable", "long_reskill"}
        for persona in ALL_PERSONAS:
            for onet_code in PROFESSIONS_100:
                score = _score_profession(self.scorer, onet_code, persona)
                assert score.bucket in valid_buckets, (
                    f"Invalid bucket '{score.bucket}' for {onet_code}"
                )

    def test_determinism(self):
        """Same inputs must always produce same outputs."""
        codes = list(PROFESSIONS_100.keys())[:10]
        for onet_code in codes:
            s1 = _score_profession(self.scorer, onet_code, ALL_PERSONAS[0])
            s2 = _score_profession(self.scorer, onet_code, ALL_PERSONAS[0])
            assert s1.match_score == s2.match_score
            assert s1.gap_severity == s2.gap_severity
            assert s1.bucket == s2.bucket

    def test_explanations_non_empty(self):
        """Every score must have a non-empty explanation and training_suggestion."""
        for persona in ALL_PERSONAS:
            for onet_code in PROFESSIONS_100:
                score = _score_profession(self.scorer, onet_code, persona)
                assert isinstance(score.explanation, str) and len(score.explanation) > 0
                assert isinstance(score.training_suggestion, str) and len(score.training_suggestion) > 0


class TestSectorCoverage:
    """Verify sector coverage and data quality."""

    def test_all_sectors_represented(self):
        """All 22 SOC prefix sectors must be covered."""
        found_prefixes = {code[:2] for code in PROFESSIONS_100}
        expected = set(SOC_SECTORS.keys())
        missing = expected - found_prefixes
        assert not missing, f"Missing sectors: {missing}"

    def test_professions_count_is_100(self):
        """Exactly 100 professions must be defined."""
        assert len(PROFESSIONS_100) == 100, (
            f"Expected 100 professions, got {len(PROFESSIONS_100)}"
        )

    def test_job_zone_diversity(self):
        """All 5 job zones (1-5) must be represented."""
        zones = {p["job_zone"] for p in PROFESSIONS_100.values()}
        for z in range(1, 6):
            assert z in zones, f"Job zone {z} not represented"

    def test_skill_count_reasonable(self):
        """Each profession must have 5-10 skills."""
        for code, prof in PROFESSIONS_100.items():
            n = len(prof["skills"])
            assert 5 <= n <= 10, (
                f"{code} ({prof['title']}) has {n} skills, expected 5-10"
            )

    def test_each_sector_has_multiple_professions(self):
        """Each sector should have at least 4 professions."""
        from collections import Counter
        prefix_counts = Counter(code[:2] for code in PROFESSIONS_100)
        for prefix, count in prefix_counts.items():
            assert count >= 4, (
                f"Sector {prefix} ({SOC_SECTORS.get(prefix, '?')}) has only {count} professions"
            )


class TestCrossSectorTransitions:
    """Test transitions across different sectors."""

    def setup_method(self):
        self.scorer = BaselineScorer()

    def test_all_buckets_reachable(self):
        """At least one profession must fall into each bucket across all personas."""
        buckets_seen = set()
        for persona in ALL_PERSONAS:
            for onet_code in PROFESSIONS_100:
                score = _score_profession(self.scorer, onet_code, persona)
                buckets_seen.add(score.bucket)
        assert "ready_now" in buckets_seen, "No profession scored ready_now"
        assert "trainable" in buckets_seen, "No profession scored trainable"
        assert "long_reskill" in buckets_seen, "No profession scored long_reskill"

    def test_tech_worker_scores_higher_in_tech_sector(self):
        """Tech worker should generally score higher on tech jobs than distant sectors."""
        tech_codes = [c for c in PROFESSIONS_100 if c.startswith("15")]
        farm_codes = [c for c in PROFESSIONS_100 if c.startswith("45")]
        tech_scores = [
            _score_profession(self.scorer, c, PERSONA_TECH_WORKER).match_score
            for c in tech_codes
        ]
        farm_scores = [
            _score_profession(self.scorer, c, PERSONA_TECH_WORKER).match_score
            for c in farm_codes
        ]
        avg_tech = sum(tech_scores) / len(tech_scores)
        avg_farm = sum(farm_scores) / len(farm_scores)
        assert avg_tech > avg_farm, (
            f"Tech worker avg tech score ({avg_tech:.1f}) should beat farm score ({avg_farm:.1f})"
        )

    def test_healthcare_worker_scores_higher_in_healthcare(self):
        """Healthcare worker should score higher on healthcare than construction."""
        health_codes = [c for c in PROFESSIONS_100 if c.startswith("29")]
        const_codes = [c for c in PROFESSIONS_100 if c.startswith("47")]
        health_scores = [
            _score_profession(self.scorer, c, PERSONA_HEALTHCARE_WORKER).match_score
            for c in health_codes
        ]
        const_scores = [
            _score_profession(self.scorer, c, PERSONA_HEALTHCARE_WORKER).match_score
            for c in const_codes
        ]
        avg_health = sum(health_scores) / len(health_scores)
        avg_const = sum(const_scores) / len(const_scores)
        assert avg_health > avg_const, (
            f"Healthcare avg ({avg_health:.1f}) should beat construction ({avg_const:.1f})"
        )

    def test_blue_collar_scores_higher_in_maintenance(self):
        """Blue collar worker should score higher in maintenance than legal."""
        maint_codes = [c for c in PROFESSIONS_100 if c.startswith("49")]
        legal_codes = [c for c in PROFESSIONS_100 if c.startswith("23")]
        maint_scores = [
            _score_profession(self.scorer, c, PERSONA_BLUE_COLLAR).match_score
            for c in maint_codes
        ]
        legal_scores = [
            _score_profession(self.scorer, c, PERSONA_BLUE_COLLAR).match_score
            for c in legal_codes
        ]
        avg_maint = sum(maint_scores) / len(maint_scores)
        avg_legal = sum(legal_scores) / len(legal_scores)
        assert avg_maint > avg_legal, (
            f"Blue collar maint avg ({avg_maint:.1f}) should beat legal ({avg_legal:.1f})"
        )


class TestScoringInvariants:
    """Test mathematical invariants of the scoring system."""

    def setup_method(self):
        self.scorer = BaselineScorer()

    def test_perfect_match_always_ready_now(self):
        """Rating 4 on all required skills must produce ready_now."""
        for onet_code, prof in list(PROFESSIONS_100.items())[:20]:
            perfect = {s["element_id"]: 4 for s in prof["skills"]}
            score = self.scorer.score_occupation(
                onet_code=onet_code,
                occupation_title=prof["title"],
                occupation_skills=prof["skills"],
                user_skill_ratings=perfect,
                target_job_zone=prof["job_zone"],
            )
            assert score.match_score == 100.0, (
                f"Perfect match for {onet_code} should be 100, got {score.match_score}"
            )
            assert score.gap_severity == 0.0
            assert score.bucket == "ready_now"

    def test_no_skills_always_long_reskill(self):
        """Empty user ratings must produce long_reskill."""
        for onet_code, prof in list(PROFESSIONS_100.items())[:20]:
            score = self.scorer.score_occupation(
                onet_code=onet_code,
                occupation_title=prof["title"],
                occupation_skills=prof["skills"],
                user_skill_ratings={},
                target_job_zone=prof["job_zone"],
            )
            assert score.gap_severity == 100.0
            assert score.bucket == "long_reskill"

    def test_monotonicity(self):
        """Improving a skill rating must never worsen the match score."""
        sample_codes = list(PROFESSIONS_100.keys())[:15]
        for onet_code in sample_codes:
            prof = PROFESSIONS_100[onet_code]
            base_ratings = {s["element_id"]: 1 for s in prof["skills"]}
            base_score = self.scorer.score_occupation(
                onet_code=onet_code,
                occupation_title=prof["title"],
                occupation_skills=prof["skills"],
                user_skill_ratings=base_ratings,
                target_job_zone=prof["job_zone"],
            )
            # Improve one skill at a time
            for skill in prof["skills"]:
                improved = base_ratings.copy()
                improved[skill["element_id"]] = 3
                improved_score = self.scorer.score_occupation(
                    onet_code=onet_code,
                    occupation_title=prof["title"],
                    occupation_skills=prof["skills"],
                    user_skill_ratings=improved,
                    target_job_zone=prof["job_zone"],
                )
                assert improved_score.match_score >= base_score.match_score, (
                    f"Improving {skill['skill_name']} worsened score for {onet_code}: "
                    f"{base_score.match_score} -> {improved_score.match_score}"
                )

    def test_gap_severity_decreases_with_better_ratings(self):
        """Higher ratings must produce lower or equal gap severity."""
        sample_codes = list(PROFESSIONS_100.keys())[:15]
        for onet_code in sample_codes:
            prof = PROFESSIONS_100[onet_code]
            low_ratings = {s["element_id"]: 0 for s in prof["skills"]}
            high_ratings = {s["element_id"]: 3 for s in prof["skills"]}
            low_score = self.scorer.score_occupation(
                onet_code=onet_code,
                occupation_title=prof["title"],
                occupation_skills=prof["skills"],
                user_skill_ratings=low_ratings,
                target_job_zone=prof["job_zone"],
            )
            high_score = self.scorer.score_occupation(
                onet_code=onet_code,
                occupation_title=prof["title"],
                occupation_skills=prof["skills"],
                user_skill_ratings=high_ratings,
                target_job_zone=prof["job_zone"],
            )
            assert high_score.gap_severity <= low_score.gap_severity, (
                f"Higher ratings should reduce gap for {onet_code}: "
                f"low={low_score.gap_severity}, high={high_score.gap_severity}"
            )


class TestPerSectorSampling:
    """Parametrized tests sampling one profession per sector."""

    def setup_method(self):
        self.scorer = BaselineScorer()

    @pytest.mark.parametrize("sector_prefix", sorted(SOC_SECTORS.keys()))
    def test_sector_scoring_valid(self, sector_prefix):
        """Score one profession from each sector with each persona."""
        sector_codes = [c for c in PROFESSIONS_100 if c.startswith(sector_prefix)]
        assert len(sector_codes) > 0, f"No professions for sector {sector_prefix}"
        onet_code = sector_codes[0]
        for persona in ALL_PERSONAS:
            score = _score_profession(self.scorer, onet_code, persona)
            assert 0 <= score.match_score <= 100
            assert 0 <= score.gap_severity <= 100
            assert score.bucket in {"ready_now", "trainable", "long_reskill"}
            assert len(score.explanation) > 0

    @pytest.mark.parametrize("sector_prefix", sorted(SOC_SECTORS.keys()))
    def test_sector_gap_identification(self, sector_prefix):
        """Verify gaps are correctly identified per sector."""
        sector_codes = [c for c in PROFESSIONS_100 if c.startswith(sector_prefix)]
        onet_code = sector_codes[0]
        prof = PROFESSIONS_100[onet_code]
        # Use empty ratings to ensure all skills are gaps
        score = self.scorer.score_occupation(
            onet_code=onet_code,
            occupation_title=prof["title"],
            occupation_skills=prof["skills"],
            user_skill_ratings={},
            target_job_zone=prof["job_zone"],
        )
        assert len(score.top_gaps) == len(prof["skills"]), (
            f"With no ratings, all {len(prof['skills'])} skills should be gaps, "
            f"but got {len(score.top_gaps)} for {onet_code}"
        )


class TestMetadata:
    """Test metadata fields in scoring output."""

    def setup_method(self):
        self.scorer = BaselineScorer()

    def test_metadata_fields_present(self):
        """All scores must include expected metadata fields."""
        for onet_code in list(PROFESSIONS_100.keys())[:10]:
            score = _score_profession(self.scorer, onet_code, ALL_PERSONAS[0])
            assert "total_skills" in score.metadata
            assert "skills_with_gaps" in score.metadata
            assert score.metadata["total_skills"] == len(PROFESSIONS_100[onet_code]["skills"])

    def test_job_zone_in_metadata(self):
        """Job zone info must appear in metadata."""
        code = list(PROFESSIONS_100.keys())[0]
        prof = PROFESSIONS_100[code]
        score = _score_profession(self.scorer, code, ALL_PERSONAS[0])
        assert score.metadata["target_job_zone"] == prof["job_zone"]


class TestTrainingSuggestions:
    """Test training suggestion generation across job zones."""

    def setup_method(self):
        self.scorer = BaselineScorer()

    def test_ready_now_suggestion_is_encouraging(self):
        """Ready_now suggestions should encourage applying."""
        for onet_code, prof in PROFESSIONS_100.items():
            perfect = {s["element_id"]: 4 for s in prof["skills"]}
            score = self.scorer.score_occupation(
                onet_code=onet_code,
                occupation_title=prof["title"],
                occupation_skills=prof["skills"],
                user_skill_ratings=perfect,
                target_job_zone=prof["job_zone"],
            )
            assert "apply" in score.training_suggestion.lower(), (
                f"Ready_now suggestion for {onet_code} should mention applying"
            )

    def test_long_reskill_mentions_significant(self):
        """Long_reskill suggestions should mention significant effort."""
        for onet_code, prof in list(PROFESSIONS_100.items())[:10]:
            score = self.scorer.score_occupation(
                onet_code=onet_code,
                occupation_title=prof["title"],
                occupation_skills=prof["skills"],
                user_skill_ratings={},
                target_job_zone=prof["job_zone"],
            )
            if score.bucket == "long_reskill":
                assert "significant" in score.training_suggestion.lower() or \
                       "reskill" in score.training_suggestion.lower(), (
                    f"Long_reskill suggestion should mention significance for {onet_code}"
                )

