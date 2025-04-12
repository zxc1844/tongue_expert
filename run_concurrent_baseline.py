#!/usr/bin/env python
"""
Script to run the concurrent baseline test for the tongue vision project.
"""

import os
import sys
import logging
from src.baseline_test import main

if __name__ == "__main__":
    # Ensure the DASHCOPE_API_KEY environment variable is set
    if "DASHCOPE_API_KEY" not in os.environ:
        api_key = input("Please enter your Alibaba Cloud Dashscope API key: ").strip()
        if api_key:
            os.environ["DASHCOPE_API_KEY"] = api_key
        else:
            print("No API key provided. Exiting.")
            sys.exit(1)
    
    # Run the main function from baseline_test.py
    main() 