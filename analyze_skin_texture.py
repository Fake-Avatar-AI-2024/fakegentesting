import numpy as np
import cv2
from deepface import DeepFace
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, Dict
from skimage.feature import local_binary_pattern
from skimage import exposure
import warnings
import argparse
import sys
from scipy.stats import wasserstein_distance

class SkinTextureAnalyzer:
    def __init__(self):
        """Initialize the skin texture analyzer with default parameters"""
        self.enforce_detection = False
        # LBP parameters
        self.n_points = 24
        self.radius = 3
        # Color space ranges for skin detection
        self.lower_hsv = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_hsv = np.array([20, 150, 255], dtype=np.uint8)
        
    def preprocess_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load and preprocess image for analysis"""
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Failed to load image: {image_path}")
                return None
                
            # Ensure image is uint8
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            
            # Try different face detection approaches
            try:
                # First attempt with OpenCV
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    face_img = img[y:y+h, x:x+w]
                else:
                    # If OpenCV fails, try DeepFace
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    face_obj = DeepFace.extract_faces(
                        img_path=rgb_img,
                        detector_backend='opencv',
                        enforce_detection=False,
                        align=True
                    )
                    if face_obj:
                        face_img = face_obj[0]['face']
                        if face_img.dtype != np.uint8:
                            face_img = (face_img * 255).astype(np.uint8)
                    else:
                        # If all detection fails, use the whole image
                        face_img = img
            except:
                # If face detection fails, use the whole image
                face_img = img
            
            # Resize to consistent size
            face_img = cv2.resize(face_img, (224, 224))
            
            # Ensure the image is in BGR format
            if len(face_img.shape) == 3 and face_img.shape[2] == 3:
                if face_img.dtype != np.uint8:
                    face_img = (face_img * 255).astype(np.uint8)
                    
            return face_img
            
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {str(e)}")
            import traceback
            print("Full error:", traceback.format_exc())
            return None
    
    def detect_skin(self, image: np.ndarray) -> np.ndarray:
        """Detect and isolate skin regions in the image"""
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create skin mask
        skin_mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        
        # Apply mask to original image
        skin_region = cv2.bitwise_and(image, image, mask=skin_mask)
        return skin_region
        
    def extract_texture_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract texture features from the image"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate Local Binary Pattern
        lbp = local_binary_pattern(gray, self.n_points, self.radius, method='uniform')
        
        # Calculate histogram of LBP
        n_bins = self.n_points + 2
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        
        # Calculate GLCM features
        glcm = exposure.equalize_hist(gray)
        
        return {
            'lbp_hist': lbp_hist,
            'glcm': glcm
        }
        
    def calculate_similarity(self, 
                           ref_features: Dict[str, np.ndarray], 
                           gen_features: Dict[str, np.ndarray]) -> float:
        """Calculate similarity score between reference and generated image features"""
        # Compare LBP histograms using Wasserstein distance
        lbp_distance = wasserstein_distance(ref_features['lbp_hist'], gen_features['lbp_hist'])
        
        # Compare GLCM features using Mean Squared Error
        glcm_distance = np.mean((ref_features['glcm'] - gen_features['glcm']) ** 2)
        
        # Combine distances (lower distance = higher similarity)
        combined_distance = (lbp_distance + glcm_distance) / 2
        
        # Convert to similarity score (0-100)
        similarity = 100 * (1 - combined_distance)
        return max(0, min(100, similarity))

    def analyze_single(self, reference_path: str, generated_path: str, threshold: float) -> Dict:
        """Analyze a single generated image against the reference"""
        # Process reference image
        ref_img = self.preprocess_image(reference_path)
        if ref_img is None:
            raise ValueError(f"Failed to process reference image: {reference_path}")
            
        # Process generated image
        gen_img = self.preprocess_image(generated_path)
        if gen_img is None:
            raise ValueError(f"Failed to process generated image: {generated_path}")
            
        # Extract features
        ref_skin = self.detect_skin(ref_img)
        gen_skin = self.detect_skin(gen_img)
        ref_features = self.extract_texture_features(ref_skin)
        gen_features = self.extract_texture_features(gen_skin)
        
        # Calculate similarity
        similarity = self.calculate_similarity(ref_features, gen_features)
        
        return {
            'image': os.path.basename(generated_path),
            'texture_similarity': similarity,
            'matches_threshold': similarity >= threshold
        }

    def analyze_batch(self, 
                 reference_path: str, 
                 generated_folder: str, 
                 results_folder: str,
                 persona_name: str,
                 threshold: float) -> pd.DataFrame:
        """Analyze skin texture features for a batch of generated images"""
        
        # Create results folder
        os.makedirs(results_folder, exist_ok=True)
        
        # Process reference image
        print(f"Processing reference image: {reference_path}")
        ref_img = self.preprocess_image(reference_path)
        if ref_img is None:
            raise ValueError(f"Failed to process reference image: {reference_path}")
            
        # Get skin region and features for reference image
        ref_skin = self.detect_skin(ref_img)
        ref_features = self.extract_texture_features(ref_skin)
        
        results = []
        # Process all generated images
        total_images = len([f for f in os.listdir(generated_folder) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        processed = 0
        
        for filename in os.listdir(generated_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                processed += 1
                print(f"\rProcessing image {processed}/{total_images}: {filename}", end='')
                
                img_path = os.path.join(generated_folder, filename)
                gen_img = self.preprocess_image(img_path)
                
                if gen_img is not None:
                    # Get skin region and features for generated image
                    gen_skin = self.detect_skin(gen_img)
                    gen_features = self.extract_texture_features(gen_skin)
                    
                    # Calculate similarity score
                    similarity = self.calculate_similarity(ref_features, gen_features)
                    
                    results.append({
                        'image': filename,
                        'texture_similarity': similarity,
                        'matches_threshold': similarity >= threshold,
                        'persona': persona_name
                    })
                else:
                    print(f"\nWarning: Could not process {filename}")
        
        print("\nProcessing complete!")
        
        if not results:
            raise ValueError("No valid images found for analysis")
            
        # Create DataFrame with results
        df = pd.DataFrame(results)
        
        # Create visualizations including the reference image comparison
        self.visualize_results(df, results_folder, reference_path)
        
        return df
    
    def visualize_results(self, df: pd.DataFrame, results_folder: str, reference_path: str):
        """Create visualizations of the texture analysis results"""
        # Create results folder if it doesn't exist
        os.makedirs(results_folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Create histogram plot
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='texture_similarity', bins=20)
        plt.axvline(x=df['texture_similarity'].mean(), color='r', linestyle='--', 
                label=f"Mean: {df['texture_similarity'].mean():.2f}")
        plt.title('Distribution of Texture Similarity Scores')
        plt.xlabel('Similarity Score')
        plt.ylabel('Count')
        plt.legend()
        
        # Save histogram
        plot_path = os.path.join(results_folder, f'texture_analysis_histogram_{timestamp}.png')
        plt.savefig(plot_path)
        plt.close()
        
        # 2. Create comparison visualization
        plt.figure(figsize=(15, 5))
        
        # Get most and least similar images
        most_similar_row = df.loc[df['texture_similarity'].idxmax()]
        least_similar_row = df.loc[df['texture_similarity'].idxmin()]
        
        # Load reference image
        ref_img = cv2.imread(reference_path)
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
        
        # Load most and least similar images
        generated_dir = os.path.dirname(os.path.dirname(results_folder)) + '/generated/' + os.path.basename(results_folder)
        most_similar_path = os.path.join(generated_dir, most_similar_row['image'])
        least_similar_path = os.path.join(generated_dir, least_similar_row['image'])
        
        most_similar_img = cv2.imread(most_similar_path)
        most_similar_img = cv2.cvtColor(most_similar_img, cv2.COLOR_BGR2RGB)
        
        least_similar_img = cv2.imread(least_similar_path)
        least_similar_img = cv2.cvtColor(least_similar_img, cv2.COLOR_BGR2RGB)
        
        # Create subplots
        plt.subplot(1, 3, 1)
        plt.imshow(ref_img)
        plt.title('Reference Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(most_similar_img)
        plt.title(f'Most Similar\nScore: {most_similar_row["texture_similarity"]:.2f}')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(least_similar_img)
        plt.title(f'Least Similar\nScore: {least_similar_row["texture_similarity"]:.2f}')
        plt.axis('off')
        
        plt.suptitle('Texture Similarity Comparison')
        
        # Save comparison plot
        comparison_path = os.path.join(results_folder, f'texture_comparison_{timestamp}.png')
        plt.savefig(comparison_path, bbox_inches='tight', pad_inches=0.2)
        plt.close()
        
        # Save results to CSV
        csv_path = os.path.join(results_folder, f'texture_analysis_{timestamp}.csv')
        df.to_csv(csv_path, index=False)      
        return None

    def print_analysis_summary(self, df: pd.DataFrame, persona_name: str):
        """Print a detailed summary of the analysis results"""
        print(f"\nSummary for {persona_name}")
        print("=" * 40)
        
        # Calculate statistics
        total_images = len(df)
        avg_similarity = df['texture_similarity'].mean()
        median_similarity = df['texture_similarity'].median()
        min_similarity = df['texture_similarity'].min()
        max_similarity = df['texture_similarity'].max()
        
        # Get top and bottom 5 images
        top_5 = df.nlargest(5, 'texture_similarity')[['image', 'texture_similarity']]
        bottom_5 = df.nsmallest(5, 'texture_similarity')[['image', 'texture_similarity']]
        
        # Print summary statistics
        print(f"Total Images: {total_images}")
        print(f"Average Similarity: {avg_similarity:.1f}%")
        print(f"Median Similarity: {median_similarity:.1f}%")
        print(f"Similarity Range: {min_similarity:.1f}% - {max_similarity:.1f}%")
        
        # Print top 5 most similar images
        print("\nTop 5 Most Similar Images:")
        for _, row in top_5.iterrows():
            print(f"  {row['image']}: {row['texture_similarity']:.1f}%")
        
        # Print bottom 5 least similar images
        print("\nBottom 5 Least Similar Images:")
        for _, row in bottom_5.iterrows():
            print(f"  {row['image']}: {row['texture_similarity']:.1f}%")
            
def find_reference_image(ref_dir: str, persona_name: str) -> str:
    """Find the reference image with any supported extension"""
    supported_extensions = ('.jpg', '.jpeg', '.png')
    
    for ext in supported_extensions:
        potential_path = os.path.join(ref_dir, f"{persona_name}{ext}")
        if os.path.isfile(potential_path):
            return potential_path
            
    # Also try looking for the file directly (in case extension is uppercase)
    for filename in os.listdir(ref_dir):
        name, ext = os.path.splitext(filename)
        if name.lower() == persona_name.lower() and ext.lower() in supported_extensions:
            return os.path.join(ref_dir, filename)

def main():
    parser = argparse.ArgumentParser(description='Analyze skin texture features in generated faces')
    
    # Main command argument
    parser.add_argument('--mode', 
                       choices=['all', 'single', 'list'],
                       default='list',
                       help='Analysis mode: analyze all personas, single persona, or list available personas')
    
    # Base folder for the analysis
    parser.add_argument('--base-dir', 
                       default='avatar_analysis',
                       help='Base directory containing reference, generated, and results folders')
    
    # Optional persona name for single analysis
    parser.add_argument('--persona', 
                       help='Name of specific persona to analyze (used with --mode single)')
    
    # Optional threshold for similarity
    parser.add_argument('--threshold',
                       type=float,
                       default=50.0,
                       help='Minimum similarity threshold (default: 50.0)')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = SkinTextureAnalyzer()
    
    # Validate base directory
    if not os.path.exists(args.base_dir):
        print(f"Error: Base directory '{args.base_dir}' does not exist.")
        sys.exit(1)
        
    ref_dir = os.path.join(args.base_dir, 'reference')
    if not os.path.exists(ref_dir):
        print(f"Error: Reference directory '{ref_dir}' does not exist.")
        sys.exit(1)
    
    # Get available personas (now using name without extension)
    available_personas = list(set([
        os.path.splitext(f)[0] for f in os.listdir(ref_dir)
        if os.path.splitext(f)[1].lower() in ('.jpg', '.jpeg', '.png')
    ]))
    
    if not available_personas:
        print("Error: No reference images found.")
        sys.exit(1)
    
    if args.mode == 'list':
        print("\nAvailable personas:")
        for idx, persona in enumerate(available_personas, 1):
            print(f"{idx}. {persona}")
        
        while True:
            try:
                choice = input("\nEnter persona number to analyze (or 'all' for all personas): ")
                if choice.lower() == 'all':
                    args.mode = 'all'
                    break
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(available_personas):
                    args.mode = 'single'
                    args.persona = available_personas[choice_idx]
                    break
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number or 'all'.")
    
    if args.mode == 'single':
        if not args.persona or args.persona not in available_personas:
            print(f"Error: Invalid persona specified.")
            sys.exit(1)
            
        print(f"\nAnalyzing persona: {args.persona}")
        
        # Find reference image with proper extension
        reference_path = find_reference_image(ref_dir, args.persona)
        if not reference_path:
            print(f"Error: Could not find reference image for {args.persona}")
            print(f"Looked in: {ref_dir}")
            print("Available files:", os.listdir(ref_dir))
            sys.exit(1)
            
        print(f"Found reference image: {reference_path}")
        if not os.path.isfile(reference_path):
            print(f"Error: Reference file exists but cannot be accessed")
            print(f"Full path: {os.path.abspath(reference_path)}")
            print(f"File exists check:", os.path.exists(reference_path))
            print(f"Is file check:", os.path.isfile(reference_path))
            sys.exit(1)
            
        generated_folder = os.path.join(args.base_dir, 'generated', args.persona)
        results_folder = os.path.join(args.base_dir, 'results', args.persona)
        
        if not os.path.exists(generated_folder):
            print(f"Error: Generated images folder not found: {generated_folder}")
            sys.exit(1)
        
        print(f"Using generated folder: {generated_folder}")
        print(f"Will save results to: {results_folder}")
            
        try:
            df = analyzer.analyze_batch(
                reference_path=reference_path,
                generated_folder=generated_folder,
                results_folder=results_folder,
                persona_name=args.persona,
                threshold=args.threshold
            )
            
            # Print summary
            analyzer.print_analysis_summary(df, args.persona)
            
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            import traceback
            print("Full error:", traceback.format_exc())
            sys.exit(1)

if __name__ == "__main__":
    main()