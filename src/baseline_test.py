import os
import json
import base64
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from tqdm import tqdm
import openai
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class TongueVisionTest:
    """Class for testing the VL-MAX model on tongue images"""
    
    def __init__(self, data_dir="data/TonguExpertDatabase", output_dir="out_put/baseline_results"):
        """
        Initialize the tester
        
        Args:
            data_dir (str): Path to the data directory
            output_dir (str): Path to the output directory
        """
        self.data_dir = Path(data_dir)
        logger.info(f"Using data directory: {self.data_dir.absolute()}")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Using output directory: {self.output_dir.absolute()}")
        
        self.phenotypes_dir = self.data_dir / "Phenotypes"
        self.images_dir = self.data_dir / "TongueImage" / "Raw"
        
        # Check if directories exist
        if not self.data_dir.exists():
            logger.error(f"Data directory does not exist: {self.data_dir.absolute()}")
            raise FileNotFoundError(f"Data directory does not exist: {self.data_dir.absolute()}")
            
        if not self.phenotypes_dir.exists():
            logger.error(f"Phenotypes directory does not exist: {self.phenotypes_dir.absolute()}")
            raise FileNotFoundError(f"Phenotypes directory does not exist: {self.phenotypes_dir.absolute()}")
            
        if not self.images_dir.exists():
            logger.error(f"Images directory does not exist: {self.images_dir.absolute()}")
            raise FileNotFoundError(f"Images directory does not exist: {self.images_dir.absolute()}")
        
        # Initialize OpenAI API for Dashscope
        self.api_key = os.environ.get("DASHCOPE_API_KEY")
        if not self.api_key:
            logger.error("DASHCOPE_API_KEY environment variable not set.")
            raise ValueError("DASHCOPE_API_KEY environment variable not set.")
        
        # Initialize OpenAI client with Alibaba Cloud Dashscope base URL
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        # Data structures
        self.labels_df = None  # To store ground truth labels
        self.image_paths = {}  # Map SID to image path
        self.predictions = []  # To store predictions
        self.results = {}  # To store evaluation results
        
    def load_data(self):
        """Load labels and image paths"""
        logger.info("Loading labels and image paths...")
        
        # Load labels
        labels_path = self.phenotypes_dir / "L2_Labels_Predict.txt"
        if not labels_path.exists():
            logger.error(f"Labels file does not exist: {labels_path}")
            raise FileNotFoundError(f"Labels file does not exist: {labels_path}")
        
        self.labels_df = pd.read_csv(labels_path, sep='\t')
        logger.info(f"Loaded {len(self.labels_df)} labels.")
        
        # Get image paths
        image_files = list(self.images_dir.glob("*.*"))
        logger.info(f"Found {len(image_files)} image files.")
        
        # Map SID to image path
        for img_path in image_files:
            # Extract SID from filename (assuming the SID is the filename without extension)
            sid = img_path.stem
            self.image_paths[sid] = img_path
        
        # Check if all SIDs in the labels file have corresponding images
        missing_images = 0
        for sid in self.labels_df['SID']:
            if sid not in self.image_paths:
                missing_images += 1
                logger.warning(f"No image found for SID: {sid}")
        
        if missing_images > 0:
            logger.warning(f"Missing images for {missing_images} SIDs.")
        
        logger.info("Data loading completed.")
    
    def encode_image_to_base64(self, image_path):
        """
        Encode an image as base64
        
        Args:
            image_path (Path): Path to the image file
            
        Returns:
            str: Base64 encoded image
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def call_vision_model(self, image_path, model_name="qwen-vl-max"):
        """
        Call the Tongyi Qianwen VL-MAX model using OpenAI's compatible interface
        
        Args:
            image_path (Path): Path to the image file
            model_name (str): Name of the model to use
            
        Returns:
            str: The model's response
        """
        try:
            # Encode the image as base64
            base64_image = self.encode_image_to_base64(image_path)
            
            # Make the API call
            response = self.client.chat.completions.create(
                model=model_name,
               messages = [
    {
        "role": "system",
        "content": "You are an expert in Traditional Chinese Medicine (TCM) specializing in tongue diagnosis. Your task is to analyze the provided tongue image and classify specific features according to the predefined English labels. You MUST adhere strictly to the provided label options and output format. Output ONLY the JSON object."
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": """Analyze the tongue image provided and classify the following five indicators. For each indicator, select ONLY ONE label from the exact options provided below.

1.  **coating_label** (Tongue coating characteristic):
    Options: [`greasy`, `greasy_thick`, `non_greasy`]

2.  **tai_label** (Color of tongue coating):
    Options: [`white`, `light_yellow`, `yellow`]

3.  **zhi_label** (Color of tongue body):
    Options: [`regular`, `dark`, `light`]

4.  **fissure_label** (Cracks on tongue):
    Options: [`NaN`, `light`, `severe`]
    (Use `NaN` if no fissures are visible.)

5.  **tooth_mk_label** (Tooth marks on tongue sides):
    Options: [`NaN`, `light`, `severe`]
    (Use `NaN` if no tooth marks are visible.)

Your entire response must be ONLY a single JSON object containing these five keys and their corresponding selected labels. Do not include any introductory text, explanations, or markdown formatting outside the JSON structure.

Example Format:
```json
{"coating_label": "greasy", "tai_label": "white", "zhi_label": "regular", "fissure_label": "NaN", "tooth_mk_label": "light"}
```"""
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            }
        ]
    }
],
                max_tokens=500,
                temperature=0.2,  # Lower temperature for more consistent results
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"API call failed: {e}")
            return None
    
    def extract_predictions(self, raw_response):
        """
        Extract predictions from the model's response
        
        Args:
            raw_response (str): Raw response from the model
            
        Returns:
            dict: Extracted predictions or None if extraction failed
        """
        try:
            # Try to find JSON in the response
            response_text = raw_response.strip()
            
            # Look for JSON object in the response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                # Parse the JSON
                predictions = json.loads(json_str)
                
                # Ensure all keys are present
                required_keys = ["coating_label", "tai_label", "zhi_label", "fissure_label", "tooth_mk_label"]
                for key in required_keys:
                    if key not in predictions:
                        logger.warning(f"Key '{key}' not found in predictions.")
                        predictions[key] = None
                
                return predictions
            else:
                logger.warning("No JSON object found in the response.")
                return None
                
        except Exception as e:
            logger.error(f"Failed to extract predictions: {e}")
            return None
    
    def run_evaluation(self, sample_limit=None):
        """
        Run the evaluation on the dataset
        
        Args:
            sample_limit (int, optional): Limit the number of samples to process (for testing)
        """
        logger.info("Starting evaluation...")
        
        # Load data if not loaded yet
        if self.labels_df is None:
            self.load_data()
        
        # Select samples to evaluate
        if sample_limit is not None and sample_limit < len(self.labels_df):
            sample_indices = np.random.choice(len(self.labels_df), sample_limit, replace=False)
            eval_df = self.labels_df.iloc[sample_indices].copy()
            logger.info(f"Using {sample_limit} random samples for evaluation.")
        else:
            eval_df = self.labels_df.copy()
            logger.info(f"Using all {len(eval_df)} samples for evaluation.")
        
        # Prepare results storage
        evaluation_results = []
        
        # Run evaluation
        for idx, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Processing images"):
            sid = row['SID']
            
            # Skip if image not found
            if sid not in self.image_paths:
                logger.warning(f"Skipping SID {sid}: No image found.")
                continue
            
            # Get image path
            image_path = self.image_paths[sid]
            
            # Call the model
            response = self.call_vision_model(image_path)
            
            if response is None:
                logger.warning(f"Skipping SID {sid}: Model response is None.")
                continue
            
            # Extract predictions
            predictions = self.extract_predictions(response)
            
            if predictions is None:
                logger.warning(f"Skipping SID {sid}: Failed to extract predictions.")
                continue
            
            # Store result with standardized ground truth values
            result = {
                "SID": sid,
                "ground_truth": {
                    "coating_label": self.standardize_label(row["coating_label"] if not pd.isna(row["coating_label"]) else None),
                    "tai_label": self.standardize_label(row["tai_label"] if not pd.isna(row["tai_label"]) else None),
                    "zhi_label": self.standardize_label(row["zhi_label"] if not pd.isna(row["zhi_label"]) else None),
                    "fissure_label": self.standardize_label(row["fissure_label"] if not pd.isna(row["fissure_label"]) else None),
                    "tooth_mk_label": self.standardize_label(row["tooth_mk_label"] if not pd.isna(row["tooth_mk_label"]) else None)
                },
                "predictions": {
                    "coating_label": self.standardize_label(predictions["coating_label"]),
                    "tai_label": self.standardize_label(predictions["tai_label"]),
                    "zhi_label": self.standardize_label(predictions["zhi_label"]),
                    "fissure_label": self.standardize_label(predictions["fissure_label"]),
                    "tooth_mk_label": self.standardize_label(predictions["tooth_mk_label"])
                },
                "raw_response": response
            }
            
            evaluation_results.append(result)
        
        self.predictions = evaluation_results
        logger.info(f"Evaluation completed. Processed {len(evaluation_results)} images.")
    
    def standardize_label(self, value):
        """
        Standardize label values to handle different representations of missing/null values
        
        Args:
            value: The label value to standardize
            
        Returns:
            str: Standardized label value
        """
        # Convert to string for consistent comparison
        if value is None:
            return "NaN"  # Use "NaN" as the standard representation for null/None
        
        # Convert to string and handle other variations
        value_str = str(value)
        
        # Handle various null-like representations
        if value_str.lower() in ["nan", "none", "null"]:
            return "NaN"
            
        return value_str
    
    def calculate_metrics(self):
        """Calculate evaluation metrics"""
        logger.info("Calculating metrics...")
        
        if not self.predictions:
            logger.error("No predictions available. Run evaluation first.")
            return
        
        # Prepare data for metric calculation
        metrics = {}
        indicators = ["coating_label", "tai_label", "zhi_label", "fissure_label", "tooth_mk_label"]
        
        for indicator in indicators:
            y_true = []
            y_pred = []
            
            for result in self.predictions:
                ground_truth = result["ground_truth"][indicator]
                prediction = result["predictions"][indicator]
                
                # Standardize both values for proper comparison
                standardized_truth = self.standardize_label(ground_truth)
                standardized_pred = self.standardize_label(prediction)
                
                # Add to lists for metrics calculation
                y_true.append(standardized_truth)
                y_pred.append(standardized_pred)
            
            if len(y_true) > 0:
                try:
                    # Calculate metrics with zero_division=0 to avoid warnings
                    metrics[indicator] = {
                        "accuracy": accuracy_score(y_true, y_pred),
                        "sample_count": len(y_true),
                        "detailed_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                    }
                    
                    # Calculate these metrics with zero_division=0 to avoid warnings
                    metrics[indicator]["precision_macro"] = precision_score(y_true, y_pred, average='macro', zero_division=0)
                    metrics[indicator]["recall_macro"] = recall_score(y_true, y_pred, average='macro', zero_division=0)
                    metrics[indicator]["f1_macro"] = f1_score(y_true, y_pred, average='macro', zero_division=0)
                except Exception as e:
                    logger.warning(f"Could not calculate metrics for {indicator}: {e}")
                    metrics[indicator] = {
                        "accuracy": None,
                        "sample_count": len(y_true),
                        "error": str(e)
                    }
            else:
                metrics[indicator] = {
                    "accuracy": None,
                    "sample_count": 0,
                    "note": "No valid predictions found"
                }
        
        # Calculate overall accuracy (if all indicators are correct for a sample, it's counted as correct)
        overall_correct = 0
        overall_total = 0
        
        for result in self.predictions:
            all_correct = True
            
            for indicator in indicators:
                ground_truth = result["ground_truth"][indicator]
                prediction = result["predictions"][indicator]
                
                # Standardize both values for proper comparison
                standardized_truth = self.standardize_label(ground_truth)
                standardized_pred = self.standardize_label(prediction)
                
                if standardized_truth != standardized_pred:
                    all_correct = False
                    break
            
            overall_total += 1
            if all_correct:
                overall_correct += 1
        
        if overall_total > 0:
            metrics["overall"] = {
                "accuracy": overall_correct / overall_total,
                "sample_count": overall_total
            }
        else:
            metrics["overall"] = {
                "accuracy": None,
                "sample_count": 0,
                "note": "No valid samples found"
            }
        
        self.results = metrics
        logger.info("Metrics calculation completed.")
    
    def save_results(self):
        """Save evaluation results and metrics to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw predictions
        predictions_file = self.output_dir / f"predictions_{timestamp}.json"
        with open(predictions_file, 'w', encoding='utf-8') as f:
            json.dump(self.predictions, f, ensure_ascii=False, indent=2)
        logger.info(f"Predictions saved to: {predictions_file}")
        
        # Save metrics
        metrics_file = self.output_dir / f"metrics_{timestamp}.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        logger.info(f"Metrics saved to: {metrics_file}")
        
        # Generate a human-readable report
        report_file = self.output_dir / f"report_{timestamp}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Tongue Vision Model Baseline Test Report\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Overall Performance\n\n")
            overall = self.results.get("overall", {})
            if overall.get("accuracy") is not None:
                f.write(f"Overall Accuracy: {overall['accuracy']:.4f}\n")
                f.write(f"Sample Count: {overall['sample_count']}\n\n")
            else:
                f.write("No valid samples found for overall accuracy calculation.\n\n")
            
            f.write("## Performance by Indicator\n\n")
            for indicator in ["coating_label", "tai_label", "zhi_label", "fissure_label", "tooth_mk_label"]:
                metrics = self.results.get(indicator, {})
                f.write(f"### {indicator}\n\n")
                
                if metrics.get("accuracy") is not None:
                    f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
                    f.write(f"Sample Count: {metrics['sample_count']}\n")
                    
                    if "precision_macro" in metrics:
                        f.write(f"Precision (Macro): {metrics['precision_macro']:.4f}\n")
                        f.write(f"Recall (Macro): {metrics['recall_macro']:.4f}\n")
                        f.write(f"F1 Score (Macro): {metrics['f1_macro']:.4f}\n")
                    
                    f.write("\nDetailed Classification Report:\n\n")
                    f.write("| Class | Precision | Recall | F1 Score | Support |\n")
                    f.write("|-------|-----------|--------|----------|--------|\n")
                    
                    if "detailed_report" in metrics:
                        for class_name, class_metrics in metrics["detailed_report"].items():
                            if class_name not in ["accuracy", "macro avg", "weighted avg"]:
                                f.write(f"| {class_name} | {class_metrics['precision']:.4f} | {class_metrics['recall']:.4f} | {class_metrics['f1-score']:.4f} | {class_metrics['support']} |\n")
                else:
                    if "error" in metrics:
                        f.write(f"Error calculating metrics: {metrics['error']}\n")
                    else:
                        f.write("No valid predictions found for this indicator.\n")
                
                f.write("\n")
            
            # Add a summary of common errors (optional)
            f.write("## Common Errors\n\n")
            f.write("This section would analyze common error patterns (to be implemented).\n\n")
            
            # Add recommendations (optional)
            f.write("## Recommendations\n\n")
            f.write("Based on the analysis, the following improvements could be considered:\n\n")
            f.write("1. Further fine-tuning of the model on more tongue diagnosis images\n")
            f.write("2. Improving image preprocessing to enhance key features\n")
            f.write("3. Adjusting the prompt to provide more specific guidance\n")
        
        logger.info(f"Report saved to: {report_file}")
        
        return {
            "predictions_file": predictions_file,
            "metrics_file": metrics_file,
            "report_file": report_file
        }
    
    def run_pipeline(self, sample_limit=None):
        """Run the complete evaluation pipeline"""
        try:
            logger.info("Starting the evaluation pipeline...")
            self.load_data()
            self.run_evaluation(sample_limit)
            self.calculate_metrics()
            output_files = self.save_results()
            logger.info("Evaluation pipeline completed successfully.")
            return output_files
        except Exception as e:
            logger.error(f"Error in evaluation pipeline: {e}")
            raise

def main():
    """Main function"""
    # Check for environment variable
    if "DASHCOPE_API_KEY" not in os.environ:
        logger.error("DASHCOPE_API_KEY environment variable not set.")
        logger.error("Please set the environment variable with your Alibaba Cloud Dashscope API key.")
        return
    
    try:
        # Create the tester instance
        tester = TongueVisionTest()
        
        # Run the pipeline with a sample limit for testing (remove or set to None for full evaluation)
        # This is useful for initial testing to avoid consuming too many API credits
        sample_limit = 10  # Set to None to run on all images
        
        # Run the pipeline
        output_files = tester.run_pipeline(sample_limit)
        
        # Print the output file paths
        logger.info("Evaluation completed successfully.")
        logger.info(f"Predictions saved to: {output_files['predictions_file']}")
        logger.info(f"Metrics saved to: {output_files['metrics_file']}")
        logger.info(f"Report saved to: {output_files['report_file']}")
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 