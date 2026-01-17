#!/usr/bin/env python3
"""
Run all test suites for the ARC training pipeline.

Run with: python -m src.tests.run_all_tests
"""

import sys
import subprocess


def run_test_module(module_name: str) -> bool:
    """Run a test module and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {module_name}")
    print("=" * 60)
    
    result = subprocess.run(
        [sys.executable, "-m", module_name],
        capture_output=False
    )
    
    return result.returncode == 0


def main():
    print("=" * 60)
    print("ARC Training Pipeline - Full Test Suite")
    print("=" * 60)
    
    test_modules = [
        "src.tests.test_tokenizer",
        "src.tests.test_data_generation",
        "src.tests.test_collate_and_loss",
    ]
    
    results = []
    for module in test_modules:
        success = run_test_module(module)
        results.append((module, success))
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for module, success in results:
        status = "PASSED" if success else "FAILED"
        if not success:
            all_passed = False
        print(f"  {module}: {status}")
    
    print()
    if all_passed:
        print("ALL TEST SUITES PASSED!")
        return 0
    else:
        print("SOME TEST SUITES FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
