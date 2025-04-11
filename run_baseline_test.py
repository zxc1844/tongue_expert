#!/usr/bin/env python3
# run_baseline_test.py - Script to run the tongue vision baseline test

import os
import argparse
import logging
from dotenv import load_dotenv
from src.baseline_test import TongueVisionTest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def main():
    """Main function to run the baseline test"""
    # Load environment variables from .env file if it exists
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run baseline test for tongue vision model')
    parser.add_argument('--sample', type=int, default=10, 
                        help='Number of samples to test. Set to -1 for all images.')
    parser.add_argument('--api-key', type=str, default=None,
                        help='Alibaba Cloud Dashscope API key (overrides environment variable)')
    parser.add_argument('--model', type=str, default="qwen-vl-max",
                        help='Model name to use for the API calls')
    parser.add_argument('--base-url', type=str, 
                        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
                        help='Base URL for the API calls')
    parser.add_argument('--data-dir', type=str, default="data/TonguExpertDatabase",
                        help='Path to the data directory')
    parser.add_argument('--output-dir', type=str, default="out_put/baseline_results",
                        help='Path to the output directory')
    
    args = parser.parse_args()
    
    # Set API key from command line if provided
    if args.api_key:
        os.environ["DASHCOPE_API_KEY"] = args.api_key
    
    # Check if API key is set
    if "DASHCOPE_API_KEY" not in os.environ:
        logger.error("DASHCOPE_API_KEY environment variable not set.")
        logger.error("Please set the environment variable or provide it via --api-key.")
        return 1
    
    # Convert sample limit
    sample_limit = None if args.sample == -1 else args.sample
    
    try:
        # Create the tester instance
        tester = TongueVisionTest(data_dir=args.data_dir, output_dir=args.output_dir)
        
        # Override the client base_url if provided
        if args.base_url:
            tester.client.base_url = args.base_url
            logger.info(f"Using base URL: {args.base_url}")
        
        # Run the pipeline
        output_files = tester.run_pipeline(sample_limit)
        
        # Print the output file paths
        logger.info("Evaluation completed successfully.")
        logger.info(f"Predictions saved to: {output_files['predictions_file']}")
        logger.info(f"Metrics saved to: {output_files['metrics_file']}")
        logger.info(f"Report saved to: {output_files['report_file']}")
        
        return 0
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main()) 