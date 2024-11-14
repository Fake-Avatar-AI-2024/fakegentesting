import numpy as np
from deepface import DeepFace
import pandas as pd
import os, sys, argparse, warnings
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional

class FaceNet512Analyzer:
    def __init__(self):
        """Initialize FaceNet512 analyzer"""
        self.model_name = "Facenet512"
        self.enforce_detection = False
        
    def extract_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """Extract facial embedding using FaceNet512"""
        try:
            embedding_dict = DeepFace.represent(
                img_path=image_path,
                model_name=self.model_name,
                enforce_detection=self.enforce_detection
            )
            
            #Handle both single dict and list of dicts
            if isinstance(embedding_dict, list):
                if not embedding_dict: # Empty list
                    return None
                embedding_dict = embedding_dict[0]
            
            #Extract embedding array from dictionary
            embedding = embedding_dict.get('embedding',None)
            if embedding is None:
                print(f"No embedding in result for {image_path}")
                return None
            
            return np.array(embedding)
        
        except Exception as e:
            print(f"Error extracting embedding from {image_path}: {str(e)}")
            return None

    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate similarity score between two embeddings"""
        if embedding1 is None or embedding2 is None:
            return 0.0
            
        embedding1 = embedding1.flatten()
        embedding2 = embedding2.flatten()
        
        if embedding1.size == 0 or embedding2.size == 0:
            return 0.0
        
        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        
        return (similarity + 1) / 2 * 100

    def analyze_persona(self, 
                       reference_path: str, 
                       generated_folder: str, 
                       results_folder: str,
                       persona_name: Optional[str] = None) -> pd.DataFrame:
        """Analyze generated images for a single persona"""
        
        # Create results folder
        os.makedirs(results_folder, exist_ok=True)
        
        # Get reference embedding
        print(f"Processing reference image: {reference_path}")
        ref_embedding = self.extract_embedding(reference_path)
        if ref_embedding is None:
            raise ValueError(f"Failed to process reference image: {reference_path}")
        
        results = []
        # Process all generated images
        total_images = len([f for f in os.listdir(generated_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        processed = 0
        
        for filename in os.listdir(generated_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                processed += 1
                print(f"\rProcessing {processed}/{total_images} images", end='')
                img_path = os.path.join(generated_folder, filename)
                gen_embedding = self.extract_embedding(img_path)
                
                if gen_embedding is not None:
                    similarity = self.calculate_similarity(ref_embedding, gen_embedding)
                    results.append({
                        'image': filename,
                        'similarity': similarity,
                        'persona': persona_name if persona_name else 'default'
                    })
                else:
                    print(f"Warning: Could not process {filename}")
        
        print()
        
        if not results:
            raise ValueError("No valid images found for analysis")
            
        return pd.DataFrame(results)

    def create_analysis_plots(self, 
                            df: pd.DataFrame, 
                            results_folder: str, 
                            persona_name: Optional[str] = None):
        """Create and save analysis visualizations"""
        
        # Distribution plot
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='similarity', bins=20)
        title = f'Similarity Distribution - {persona_name}' if persona_name else 'Similarity Distribution'
        plt.title(title)
        plt.xlabel('Similarity to Reference (%)')
        plt.ylabel('Count')
        
        plot_name = f'distribution_{persona_name}.png' if persona_name else 'distribution.png'
        plt.savefig(os.path.join(results_folder, plot_name))
        plt.close()
        
        # Optional: Add more visualizations here if needed

    def analyze_all_personas(self, base_folder: str):
        """
        Analyze all personas in the structured folder
        
        Expected structure:
        base_folder/
        ├── reference/
        │   ├── persona1.jpg
        │   ├── persona2.jpg
        ├── generated/
        │   ├── persona1/
        │   ├── persona2/
        └── results/
        """
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        results_base = os.path.join(base_folder, 'results', timestamp)
        os.makedirs(results_base, exist_ok=True)
        
        all_results = []
        
        # Process each reference image
        ref_folder = os.path.join(base_folder, 'reference')
        for ref_file in os.listdir(ref_folder):
            if not ref_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            persona_name = os.path.splitext(ref_file)[0]
            print(f"\nProcessing persona: {persona_name}")
            
            reference_path = os.path.join(ref_folder, ref_file)
            generated_folder = os.path.join(base_folder, 'generated', persona_name)
            results_folder = os.path.join(results_base, persona_name)
            
            if not os.path.exists(generated_folder):
                print(f"Warning: No generated images folder found for {persona_name}")
                continue
                
            try:
                # Analyze this persona
                df = self.analyze_persona(
                    reference_path=reference_path,
                    generated_folder=generated_folder,
                    results_folder=results_folder,
                    persona_name=persona_name
                )
                
                # Create visualizations
                self.create_analysis_plots(df, results_folder, persona_name)
                
                # Save individual results
                df.to_csv(os.path.join(results_folder, f'analysis_{persona_name}.csv'), index=False)
                
                # Add to overall results
                all_results.append(df)
                
                # Print summary for this persona
                self._print_persona_summary(df, persona_name)
                
            except Exception as e:
                print(f"Error processing {persona_name}: {str(e)}")
                continue
        
        if all_results:
            # Combine all results and save
            combined_df = pd.concat(all_results)
            combined_df.to_csv(os.path.join(results_base, 'combined_analysis.csv'), index=False)
            
            # Create combined visualization
            self._create_combined_plot(combined_df, results_base)

    def _print_persona_summary(self, df: pd.DataFrame, persona_name: str):
        """Print summary statistics for a persona"""
        print(f"\nSummary for {persona_name}")
        print("=" * 40)
        print(f"Total Images: {len(df)}")
        print(f"Average Similarity: {df['similarity'].mean():.1f}%")
        print(f"Median Similarity: {df['similarity'].median():.1f}%")
        print(f"Similarity Range: {df['similarity'].min():.1f}% - {df['similarity'].max():.1f}%")
        
        print("\nTop 5 Most Similar Images:")
        top_5 = df.nlargest(5, 'similarity')
        for _, row in top_5.iterrows():
            print(f"  {row['image']}: {row['similarity']:.1f}%")

    def _create_combined_plot(self, df: pd.DataFrame, results_folder: str):
        """Create combined analysis plot for all personas"""
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x='persona', y='similarity')
        plt.title('Similarity Comparison Across Personas')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(results_folder, 'combined_comparison.png'))
        plt.close()


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
            
    return None

def main():
    parser = argparse.ArgumentParser(description='Analyze avatar similarity using FaceNet512')
    
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
                       default=80.0,
                       help='Minimum similarity threshold (default: 80.0)')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = FaceNet512Analyzer()
    
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
            analyzer = FaceNet512Analyzer()
            df = analyzer.analyze_persona(
                reference_path=reference_path,
                generated_folder=generated_folder,
                results_folder=results_folder,
                persona_name=args.persona
            )
            
            # Create visualizations
            analyzer.create_analysis_plots(df, results_folder, args.persona)
            
            # Print summary
            analyzer._print_persona_summary(df, args.persona)
            
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            import traceback
            print("Full error:", traceback.format_exc())
            sys.exit(1)

if __name__ == "__main__":
    main()