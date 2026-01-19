#!/usr/bin/env python3
"""Seed script to populate database with demo data.

This script uses the Celery task to warm the occupation cache with
a predefined set of occupations for development and testing.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.tasks.tasks import warm_occupation_cache

# Predefined set of tech occupations for demo
DEMO_OCCUPATIONS = [
    "15-1252.00",  # Software Developers
    "15-1299.08",  # Web Developers
    "15-1244.00",  # Network and Computer Systems Administrators
    "15-1299.09",  # Web and Digital Interface Designers
    "15-1212.00",  # Information Security Analysts
    "11-3021.00",  # Computer and Information Systems Managers
    "15-1241.00",  # Computer Network Architects
    "15-1299.02",  # Geographic Information Systems Technologists
    "15-1211.00",  # Computer Systems Analysts
    "15-1232.00",  # Computer User Support Specialists
]


def main():
    """Run seed script."""
    print("=" * 60)
    print("SkillSprout Demo Data Seeding")
    print("=" * 60)
    print()
    print(f"This will cache {len(DEMO_OCCUPATIONS)} occupations and their skills.")
    print("Depending on O*NET availability, this may take 1-2 minutes.")
    print()

    confirmation = input("Continue? (y/n): ")
    if confirmation.lower() != 'y':
        print("Cancelled.")
        return

    print()
    print("Starting cache warming...")
    print()

    try:
        result = warm_occupation_cache(DEMO_OCCUPATIONS)
        print()
        print("=" * 60)
        print("Seeding Complete!")
        print("=" * 60)
        print(f"✓ Successfully cached: {result['cached']} occupations")
        print(f"✗ Failed: {result['failed']} occupations")
        print()
        print("You can now:")
        print("  1. Start the API: uvicorn app.main:app --reload")
        print("  2. Visit http://localhost:8000")
        print("  3. Create a user profile and start exploring!")
        print()

    except Exception as e:
        print()
        print(f"❌ Error: {e}")
        print()
        print("Make sure:")
        print("  1. Database is running and migrations are applied")
        print("  2. Redis is running (for Celery)")
        print("  3. O*NET credentials are configured (or DEMO_MODE=true)")
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()
