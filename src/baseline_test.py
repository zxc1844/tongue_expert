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
import concurrent.futures
import time
import queue
from threading import Lock
import random
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class RateLimiter:
    """A rate limiter to prevent exceeding API limits"""
    
    def __init__(self, max_calls_per_second=1, max_concurrent_requests=5):
        """
        Initialize the rate limiter
        
        Args:
            max_calls_per_second (int): Maximum number of calls allowed per second
            max_concurrent_requests (int): Maximum number of concurrent requests
        """
        self.max_calls_per_second = max_calls_per_second
        self.max_concurrent_requests = max_concurrent_requests
        self.call_timestamps = queue.Queue()
        self.lock = Lock()
        self.active_requests = 0
        
    def acquire(self):
        """
        Acquires permission to make an API call, blocking if necessary.
        """
        with self.lock:
            # Wait until we have a free slot for concurrent requests
            while self.active_requests >= self.max_concurrent_requests:
                time.sleep(0.1)
            
            # Enforce the rate limit based on calls per second
            now = time.time()
            
            # If we've made max_calls_per_second calls in the last second, wait
            if self.call_timestamps.qsize() >= self.max_calls_per_second:
                oldest_timestamp = self.call_timestamps.get()
                time_since_oldest = now - oldest_timestamp
                
                if time_since_oldest < 1.0:
                    # Sleep to respect the rate limit
                    sleep_time = 1.0 - time_since_oldest + random.uniform(0.1, 0.3)  # Add jitter
                    time.sleep(sleep_time)
            
            # Record this call timestamp
            self.call_timestamps.put(time.time())
            self.active_requests += 1
    
    def release(self):
        """
        Releases a request slot.
        """
        with self.lock:
            self.active_requests -= 1

class TongueVisionTest:
    """Class for testing the VL-MAX model on tongue images"""
    
    def __init__(self, data_dir="data/TonguExpertDatabase", output_dir="out_put/baseline_results", model_name="qwen-vl-max"):
        """
        Initialize the tester
        
        Args:
            data_dir (str): Path to the data directory
            output_dir (str): Path to the output directory
            model_name (str): Name of the model to use for API calls
        """
        self.data_dir = Path(data_dir)
        logger.info(f"Using data directory: {self.data_dir.absolute()}")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Using output directory: {self.output_dir.absolute()}")
        
        self.phenotypes_dir = self.data_dir / "Phenotypes"
        self.images_dir = self.data_dir / "TongueImage" / "Raw"
        
        # Store the model name
        self.model_name = model_name
        logger.info(f"Using model: {self.model_name}")
        
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
    
    def call_vision_model(self, image_path):
        """
        Call the Tongyi Qianwen VL-MAX model using OpenAI's compatible interface
        
        Args:
            image_path (Path): Path to the image file
            
        Returns:
            str: The model's response
        """
        try:
            # Encode the image as base64
            base64_image = self.encode_image_to_base64(image_path)
            
            # Make the API call
            response = self.client.chat.completions.create(
                model=self.model_name,
              messages = [
    {
        "role": "system",
        "content": "你是一位经验丰富的中医（老中医），尤其擅长舌诊。你的任务是运用你专业的视觉判断标准，仔细分析提供的舌头图像，并根据中医（TCM）的经典视觉特征对舌象进行分类。请严格专注于图像本身的视觉信息，并严格遵循下面提供的标签选项和输出格式要求。最终仅输出JSON对象。"
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": """请根据中医舌诊的视觉判断标准，仔细分析下图，并对以下五个指标进行分类。对于每个指标，请从下面提供的选项中选择**唯一一个**最符合图像视觉特征的**英文标签**。括号中的中文描述了该英文标签对应的中医视觉标准，请以此作为你这位老中医进行视觉判断的核心依据。

1.  **`coating_label` (舌苔的质地视觉特征):**
    选项 (Options): [
      `greasy` (视觉上，苔质颗粒细腻致密、或伴有黏液、显得油亮、不清爽。对应中医【腻苔】的典型视觉),
      `greasy_thick` (视觉上，苔质特征同'greasy'，但明显更厚、更密实、刮之不去感更强。对应中医【厚腻苔】的典型视觉),
      `non_greasy` (视觉上，苔质不具备'greasy'的油腻、黏厚感，可能呈现薄、净、颗粒相对清晰或略干的状态。对应中医视觉上非腻苔，如薄苔、正常苔等的质地)
    ]

2.  **`tai_label` (舌苔的主要颜色视觉特征):**
    选项 (Options): [
      `white` (视觉上，舌苔整体呈现清晰的白色。对应中医【白苔】的视觉),
      `light_yellow` (视觉上，舌苔整体呈现淡淡的、浅浅的黄色调。对应中医【淡黄苔/薄黄苔】的视觉),
      `yellow` (视觉上，舌苔整体呈现明显、较深的黄色调。对应中医【黄苔】的视觉)
    ]

3.  **`zhi_label` (舌头的本体颜色视觉特征，即舌质颜色):**
    选项 (Options): [
      `regular` (视觉上，舌体颜色是健康的淡红色或鲜活的粉红色。对应中医【淡红舌】的标准视觉),
      `dark` (视觉上，舌体颜色明显深红、暗红、绛红，或呈现明显的紫色、青紫色。对应中医【红绛舌/紫暗舌】等的视觉),
      `light` (视觉上，舌体颜色明显浅淡、发白，缺乏红润光泽，呈"缺血"外观。对应中医【淡白舌】的视觉)
    ]

4.  **`fissure_label` (舌面上的裂纹视觉特征):**
    选项 (Options): [
      `NaN` (视觉上，舌面上完全没有裂纹或明显的沟壑。对应中医【无裂纹】的视觉),
      `light` (视觉上，舌面可见少量裂纹，或裂纹形态较浅、较细。对应中医【少许/浅裂纹】的视觉),
      `severe` (视觉上，舌面可见较多裂纹，或裂纹形态明显较深、较粗、范围较广。对应中医【多/深裂纹】的视觉)
    ]
    (如果视觉上完全看不到裂纹，请选择 `NaN`。)

5.  **`tooth_mk_label` (舌头边缘的齿痕视觉特征):**
    选项 (Options): [
      `NaN` (视觉上，舌头边缘光滑或形态自然，没有牙齿压迫形成的印痕。对应中医【无齿痕】的视觉),
      `light` (视觉上，舌头边缘可见轻微的、较浅的波浪状压痕。对应中医【轻微/浅齿痕】的视觉),
      `severe` (视觉上，舌头边缘可见非常明显的、较深的波浪状压痕，舌体可能显得胖大。对应中医【明显/深齿痕】的视觉)
    ]
    (如果视觉上完全看不到齿痕，请选择 `NaN`。)

请严格按照老中医的视觉判断标准进行评估。你的整个回答**必须**仅仅是一个JSON对象，其中包含这五个**英文**键（`coating_label`, `tai_label`, `zhi_label`, `fissure_label`, `tooth_mk_label`）和它们对应的、你根据视觉判断所选择的**英文**标签值。确保输出的JSON格式正确，不要包含任何括号中的中文描述或其他解释性文字。

输出格式示例 (Example Format):
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
    
    def process_image(self, item, rate_limiter=None):
        """
        Process a single image with API rate limiting
        
        Args:
            item (tuple): A tuple containing (sid, row) where sid is the image identifier
                         and row is the dataframe row with labels
            rate_limiter (RateLimiter, optional): Rate limiter instance to control API call frequency
            
        Returns:
            dict: Result dictionary or None if processing failed
        """
        sid, row = item
        
        # Skip if image not found
        if sid not in self.image_paths:
            logger.warning(f"Skipping SID {sid}: No image found.")
            return None
        
        # Get image path
        image_path = self.image_paths[sid]
        
        try:
            # Acquire permission from rate limiter if provided
            if rate_limiter:
                rate_limiter.acquire()
            
            # Call the model
            response = self.call_vision_model(image_path)
            
            if response is None:
                logger.warning(f"Skipping SID {sid}: Model response is None.")
                return None
            
            # Extract predictions
            predictions = self.extract_predictions(response)
            
            if predictions is None:
                logger.warning(f"Skipping SID {sid}: Failed to extract predictions.")
                return None
            
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
            
            return result
        
        except Exception as e:
            logger.error(f"Error processing SID {sid}: {e}")
            return None
        
        finally:
            # Release the rate limiter if provided
            if rate_limiter:
                rate_limiter.release()
    
    def run_evaluation(self, sample_limit=None, max_workers=5, max_calls_per_second=2):
        """
        Run the evaluation on the dataset with concurrent processing
        
        Args:
            sample_limit (int, optional): Limit the number of samples to process (for testing)
            max_workers (int): Maximum number of concurrent workers
            max_calls_per_second (int): Maximum API calls per second
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
        
        # Create a rate limiter
        rate_limiter = RateLimiter(
            max_calls_per_second=max_calls_per_second,
            max_concurrent_requests=max_workers
        )
        
        # Prepare items for processing
        items = [(row['SID'], row) for _, row in eval_df.iterrows()]
        total_items = len(items)
        
        logger.info(f"Processing {total_items} images with {max_workers} workers " 
                   f"and {max_calls_per_second} calls per second limit...")
        
        # Prepare results storage
        evaluation_results = []
        failed_sids = []
        
        # Create a progress bar for the entire process
        pbar = tqdm(total=total_items, desc="Processing images")
        
        # Process images using concurrent workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_sid = {
                executor.submit(self.process_image, item, rate_limiter): item[0]
                for item in items
            }
            
            # Process completed tasks as they finish
            for future in concurrent.futures.as_completed(future_to_sid):
                sid = future_to_sid[future]
                pbar.update(1)
                
                try:
                    result = future.result()
                    if result is not None:
                        evaluation_results.append(result)
                    else:
                        failed_sids.append(sid)
                except Exception as e:
                    logger.error(f"Task for SID {sid} generated an exception: {e}")
                    failed_sids.append(sid)
        
        pbar.close()
        
        # Log statistics
        success_count = len(evaluation_results)
        failure_count = len(failed_sids)
        logger.info(f"Evaluation completed. Successfully processed {success_count} images.")
        
        if failure_count > 0:
            logger.warning(f"Failed to process {failure_count} images.")
            
            # Log the first few failed SIDs as examples
            if failed_sids:
                logger.warning(f"Examples of failed SIDs: {failed_sids[:min(5, len(failed_sids))]}")
                
                # Save failed SIDs to a file for reference
                failed_file = self.output_dir / f"failed_sids_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(failed_file, 'w', encoding='utf-8') as f:
                    json.dump(failed_sids, f)
                logger.info(f"List of failed SIDs saved to: {failed_file}")
        
        self.predictions = evaluation_results
    
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
    """Main function with command-line argument parsing"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run tongue vision evaluation with concurrent API calls")
    parser.add_argument("--sample", type=int, default=10, 
                      help="Number of samples to process. Set to -1 for all samples.")
    parser.add_argument("--workers", type=int, default=5, 
                      help="Maximum number of concurrent workers")
    parser.add_argument("--rate", type=int, default=2, 
                      help="Maximum API calls per second")
    parser.add_argument("--model", type=str, default="qwen-vl-max", 
                      help="Model name to use")
    parser.add_argument("--output", type=str, default="out_put/baseline_results", 
                      help="Output directory for results")
    
    args = parser.parse_args()
    
    # Check for environment variable
    if "DASHCOPE_API_KEY" not in os.environ:
        logger.error("DASHCOPE_API_KEY environment variable not set.")
        logger.error("Please set the environment variable with your Alibaba Cloud Dashscope API key.")
        return
    
    try:
        # Create the tester instance with the specified model
        tester = TongueVisionTest(output_dir=args.output, model_name=args.model)
        
        # Determine sample limit
        sample_limit = None if args.sample < 0 else args.sample
        
        # Log the concurrency settings
        logger.info(f"Starting evaluation with settings:")
        logger.info(f"  Sample limit: {sample_limit if sample_limit is not None else 'All samples'}")
        logger.info(f"  Max workers: {args.workers}")
        logger.info(f"  API rate limit: {args.rate} calls per second")
        logger.info(f"  Model: {args.model}")
        
        # Run the evaluation with concurrent processing
        tester.load_data()
        tester.run_evaluation(
            sample_limit=sample_limit,
            max_workers=args.workers,
            max_calls_per_second=args.rate
        )
        tester.calculate_metrics()
        output_files = tester.save_results()
        
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