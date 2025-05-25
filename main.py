import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
from datetime import datetime
import glob
from pathlib import Path

class PlantGrowthTracker:
    def __init__(self, images_folder="plant_images"):
        """
        Initialize the Plant Growth Tracker
        
        Args:
            images_folder (str): Path to folder containing plant images
        """
        self.images_folder = images_folder
        self.results = []
        self.create_folder_structure()
    
    def create_folder_structure(self):
        """Create necessary folders for the project"""
        folders = [self.images_folder, "results", "visualizations"]
        for folder in folders:
            Path(folder).mkdir(exist_ok=True)
        print(f"✅ Folder structure created: {', '.join(folders)}")
    
    def extract_date_from_filename(self, filename):
        """
        Extract date from filename (assumes format: plant_YYYY-MM-DD.jpg)
        If no date found, uses file modification time
        """
        try:
            # Try to extract date from filename
            parts = filename.split('_')
            for part in parts:
                if len(part) == 10 and part.count('-') == 2:
                    return datetime.strptime(part, '%Y-%m-%d').date()
        except:
            pass
        
        # Fallback to file modification time
        filepath = os.path.join(self.images_folder, filename)
        if os.path.exists(filepath):
            timestamp = os.path.getmtime(filepath)
            return datetime.fromtimestamp(timestamp).date()
        
        return datetime.now().date()
    
    def preprocess_image(self, image):
        """
        Preprocess the image for analysis
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        # Resize image for consistent processing
        height, width = image.shape[:2]
        if height > 800 or width > 800:
            scale = 800 / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        return blurred
    
    def extract_green_mask(self, image):
        """
        Create a mask for green regions (plant areas)
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Binary mask where green areas are white
        """
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define range for green colors
        # Lower and Upper bounds for green in HSV
        lower_green1 = np.array([35, 40, 40])
        upper_green1 = np.array([85, 255, 255])
        
        # Create mask for green colors
        mask = cv2.inRange(hsv, lower_green1, upper_green1)
        
        # Remove noise with morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def calculate_bounding_box_area(self, mask):
        """
        Calculate the area of the bounding box containing all green regions
        
        Args:
            mask: Binary mask of green regions
            
        Returns:
            Bounding box area and coordinates
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0, (0, 0, 0, 0)
        
        # Find the bounding box that encompasses all contours
        all_points = np.vstack(contours)
        x, y, w, h = cv2.boundingRect(all_points)
        
        return w * h, (x, y, w, h)
    
    def count_green_pixels(self, mask):
        """
        Count green pixels and calculate ratio
        
        Args:
            mask: Binary mask of green regions
            
        Returns:
            Green pixel count and ratio
        """
        green_pixels = cv2.countNonZero(mask)
        total_pixels = mask.shape[0] * mask.shape[1]
        green_ratio = green_pixels / total_pixels
        
        return green_pixels, green_ratio
    
    def estimate_leaf_count(self, mask):
        """
        Estimate the number of distinct leaves using contour analysis
        
        Args:
            mask: Binary mask of green regions
            
        Returns:
            Estimated leaf count
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area to remove noise
        min_area = 100  # Minimum area for a leaf
        significant_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        return len(significant_contours)
    
    def analyze_brightness_contrast(self, image):
        """
        Analyze brightness and contrast as health indicators
        
        Args:
            image: Input image
            
        Returns:
            Brightness and contrast values
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate brightness (mean intensity)
        brightness = np.mean(gray)
        
        # Calculate contrast (standard deviation)
        contrast = np.std(gray)
        
        return brightness, contrast
    
    def calculate_color_health_index(self, image, mask):
        """
        Calculate a color health index based on green color distribution
        
        Args:
            image: Input image
            mask: Green regions mask
            
        Returns:
            Color health index (0-100)
        """
        if cv2.countNonZero(mask) == 0:
            return 0
        
        # Extract HSV values only from green regions
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Get HSV values where mask is white (green regions)
        green_hsv = hsv[mask > 0]
        
        if len(green_hsv) == 0:
            return 0
        
        # Analyze saturation and value (brightness) of green regions
        # Higher saturation and moderate to high value indicate healthy green
        saturation = np.mean(green_hsv[:, 1])
        value = np.mean(green_hsv[:, 2])
        
        # Normalize to 0-100 scale
        health_index = (saturation / 255 * 0.7 + value / 255 * 0.3) * 100
        
        return health_index
    
    def analyze_single_image(self, image_path):
        """
        Analyze a single plant image and extract all metrics
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing all extracted metrics
        """
        # Read and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not read image: {image_path}")
            return None
        
        processed_image = self.preprocess_image(image)
        
        # Extract green mask
        green_mask = self.extract_green_mask(processed_image)
        
        # Calculate all metrics
        bbox_area, bbox_coords = self.calculate_bounding_box_area(green_mask)
        green_pixels, green_ratio = self.count_green_pixels(green_mask)
        leaf_count = self.estimate_leaf_count(green_mask)
        brightness, contrast = self.analyze_brightness_contrast(processed_image)
        health_index = self.calculate_color_health_index(processed_image, green_mask)
        
        # Extract date from filename
        filename = os.path.basename(image_path)
        date = self.extract_date_from_filename(filename)
        
        results = {
            'filename': filename,
            'date': date.strftime('%Y-%m-%d'),
            'bbox_area': int(bbox_area),
            'green_pixels': int(green_pixels),
            'green_ratio': round(green_ratio, 4),
            'leaf_count': int(leaf_count),
            'brightness': round(brightness, 2),
            'contrast': round(contrast, 2),
            'health_index': round(health_index, 2),
            'bbox_coords': bbox_coords
        }
        
        print(f"✅ Analyzed: {filename}")
        return results
    
    def process_all_images(self):
        """
        Process all images in the images folder
        """
        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(self.images_folder, ext)))
            image_files.extend(glob.glob(os.path.join(self.images_folder, ext.upper())))
        
        if not image_files:
            print(f"❌ No images found in {self.images_folder}")
            print("📝 Please add images with extensions: .jpg, .jpeg, .png, .bmp")
            return
        
        print(f"🔍 Found {len(image_files)} images to process...")
        
        # Process each image
        for image_path in image_files:
            result = self.analyze_single_image(image_path)
            if result:
                self.results.append(result)
        
        # Sort results by date
        self.results.sort(key=lambda x: x['date'])
        
        print(f"✅ Processed {len(self.results)} images successfully!")
    
    def save_results_to_json(self):
        """
        Save results to JSON file
        """
        if not self.results:
            print("❌ No results to save!")
            return
        
        json_path = "results/growth_analysis.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"💾 Results saved to: {json_path}")
    
    def create_growth_visualizations(self):
        """
        Create comprehensive growth visualization charts
        """
        if not self.results:
            print("❌ No results to visualize!")
            return
        
        df = pd.DataFrame(self.results)
        df['date'] = pd.to_datetime(df['date'])
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Plant Growth Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Bounding Box Area over time
        axes[0, 0].plot(df['date'], df['bbox_area'], marker='o', linewidth=2, markersize=6)
        axes[0, 0].set_title('Plant Size (Bounding Box Area)', fontweight='bold')
        axes[0, 0].set_ylabel('Area (pixels²)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Green Pixel Ratio over time
        axes[0, 1].plot(df['date'], df['green_ratio'], marker='s', color='green', linewidth=2, markersize=6)
        axes[0, 1].set_title('Foliage Density (Green Pixel Ratio)', fontweight='bold')
        axes[0, 1].set_ylabel('Green Ratio')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Leaf Count over time
        axes[0, 2].plot(df['date'], df['leaf_count'], marker='^', color='darkgreen', linewidth=2, markersize=6)
        axes[0, 2].set_title('Estimated Leaf Count', fontweight='bold')
        axes[0, 2].set_ylabel('Number of Leaves')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Health Index over time
        axes[1, 0].plot(df['date'], df['health_index'], marker='D', color='orange', linewidth=2, markersize=6)
        axes[1, 0].set_title('Color Health Index', fontweight='bold')
        axes[1, 0].set_ylabel('Health Index (0-100)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. Brightness over time
        axes[1, 1].plot(df['date'], df['brightness'], marker='*', color='gold', linewidth=2, markersize=8)
        axes[1, 1].set_title('Image Brightness (Light Conditions)', fontweight='bold')
        axes[1, 1].set_ylabel('Brightness')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 6. Growth Score (Composite metric)
        # Calculate growth score as weighted combination of metrics
        max_bbox = df['bbox_area'].max() if df['bbox_area'].max() > 0 else 1
        max_green = df['green_ratio'].max() if df['green_ratio'].max() > 0 else 1
        max_leaves = df['leaf_count'].max() if df['leaf_count'].max() > 0 else 1
        
        growth_score = (
            (df['bbox_area'] / max_bbox) * 0.4 +
            (df['green_ratio'] / max_green) * 0.3 +
            (df['leaf_count'] / max_leaves) * 0.2 +
            (df['health_index'] / 100) * 0.1
        ) * 100
        
        axes[1, 2].plot(df['date'], growth_score, marker='o', color='purple', linewidth=3, markersize=6)
        axes[1, 2].set_title('Overall Growth Score', fontweight='bold')
        axes[1, 2].set_ylabel('Growth Score (0-100)')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = "visualizations/growth_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Visualization saved to: {plot_path}")
    
    def generate_delta_analysis(self):
        """
        Generate delta analysis showing changes between consecutive images
        """
        if len(self.results) < 2:
            print("❌ Need at least 2 images for delta analysis!")
            return
        
        print("\n📈 DELTA ANALYSIS - Changes Between Consecutive Images:")
        print("=" * 60)
        
        for i in range(1, len(self.results)):
            prev = self.results[i-1]
            curr = self.results[i]
            
            print(f"\n📅 {prev['date']} → {curr['date']}")
            print("-" * 40)
            
            # Calculate deltas
            bbox_delta = curr['bbox_area'] - prev['bbox_area']
            green_delta = curr['green_ratio'] - prev['green_ratio']
            leaf_delta = curr['leaf_count'] - prev['leaf_count']
            health_delta = curr['health_index'] - prev['health_index']
            
            # Format output with growth indicators
            print(f"🔲 Size Change: {bbox_delta:+d} pixels² {'📈' if bbox_delta > 0 else '📉' if bbox_delta < 0 else '➡️'}")
            print(f"🌿 Foliage Change: {green_delta:+.4f} ratio {'📈' if green_delta > 0 else '📉' if green_delta < 0 else '➡️'}")
            print(f"🍃 Leaf Change: {leaf_delta:+d} leaves {'📈' if leaf_delta > 0 else '📉' if leaf_delta < 0 else '➡️'}")
            print(f"💚 Health Change: {health_delta:+.2f} points {'📈' if health_delta > 0 else '📉' if health_delta < 0 else '➡️'}")
    
    def display_summary_report(self):
        """
        Display a comprehensive summary report
        """
        if not self.results:
            print("❌ No results to display!")
            return
        
        df = pd.DataFrame(self.results)
        
        print("\n" + "="*60)
        print("🌱 PLANT GROWTH TRACKING SUMMARY REPORT")
        print("="*60)
        
        print(f"📊 Analysis Period: {df['date'].min()} to {df['date'].max()}")
        print(f"📷 Total Images Analyzed: {len(self.results)}")
        print(f"📈 Tracking Duration: {(pd.to_datetime(df['date'].max()) - pd.to_datetime(df['date'].min())).days} days")
        
        print("\n🔍 GROWTH METRICS SUMMARY:")
        print("-" * 30)
        
        # Calculate growth metrics
        initial = self.results[0]
        final = self.results[-1] if len(self.results) > 1 else initial
        
        size_growth = ((final['bbox_area'] - initial['bbox_area']) / max(initial['bbox_area'], 1)) * 100
        foliage_growth = ((final['green_ratio'] - initial['green_ratio']) / max(initial['green_ratio'], 0.001)) * 100
        leaf_growth = final['leaf_count'] - initial['leaf_count']
        health_change = final['health_index'] - initial['health_index']
        
        print(f"📏 Size Growth: {size_growth:+.1f}% ({'Growing' if size_growth > 5 else 'Stable' if size_growth > -5 else 'Declining'})")
        print(f"🌿 Foliage Growth: {foliage_growth:+.1f}% ({'Expanding' if foliage_growth > 5 else 'Stable' if foliage_growth > -5 else 'Reducing'})")
        print(f"🍃 Leaf Development: {leaf_growth:+d} leaves ({'Positive' if leaf_growth > 0 else 'Stable' if leaf_growth == 0 else 'Concerning'})")
        print(f"💚 Health Trend: {health_change:+.1f} points ({'Improving' if health_change > 2 else 'Stable' if health_change > -2 else 'Declining'})")
        
        print("\n📈 CURRENT STATUS:")
        print("-" * 20)
        print(f"🔲 Current Size: {final['bbox_area']:,} pixels²")
        print(f"🌿 Foliage Density: {final['green_ratio']:.1%}")
        print(f"🍃 Leaf Count: {final['leaf_count']} leaves")
        print(f"💚 Health Index: {final['health_index']:.1f}/100")
        print(f"☀️ Light Conditions: {final['brightness']:.1f} ({'Good' if final['brightness'] > 100 else 'Moderate' if final['brightness'] > 50 else 'Low'})")
    
    def run_complete_analysis(self):
        """
        Run the complete plant growth analysis pipeline
        """
        print("🌱 Starting Plant Growth Analysis...")
        print("="*50)
        
        # Process all images
        self.process_all_images()
        
        if not self.results:
            return
        
        # Save results
        self.save_results_to_json()
        
        # Generate visualizations
        self.create_growth_visualizations()
        
        # Display reports
        self.display_summary_report()
        self.generate_delta_analysis()
        
        print("\n✅ Analysis Complete!")
        print(f"📁 Check the 'results' and 'visualizations' folders for output files.")


# Example usage and demonstration
if __name__ == "__main__":
    # Initialize the tracker
    tracker = PlantGrowthTracker()
    
    # Run complete analysis
    tracker.run_complete_analysis()
    
    print("\n" + "="*60)
    print("🎯 HOW TO USE THIS SYSTEM:")
    print("="*60)
    print("1. 📁 Place your plant images in the 'plant_images' folder")
    print("2. 📝 Name images like: plant_2024-01-15.jpg (or any format)")
    print("3. 🚀 Run this script to analyze all images")
    print("4. 📊 View results in 'results/growth_analysis.json'")
    print("5. 📈 Check visualizations in 'visualizations/growth_analysis.png'")
    print("\n💡 For best results:")
    print("   - Use consistent lighting and background")
    print("   - Take photos from the same angle/distance")
    print("   - Upload images chronologically")
    print("   - Include dates in filenames when possible")