#!/usr/bin/env python3
"""
Plant Growth Tracking System - Setup Script
Authors: Grishma (4SF22CS071), Shreya Shenoy (4SF22CS206)
Guide: Mrs Srividya S

This script sets up the Plant Growth Tracking system and creates sample data.
"""

import os
import subprocess
import sys
from pathlib import Path

def install_requirements():
    """Install required Python packages"""
    requirements = [
        'opencv-python>=4.5.0',
        'numpy>=1.21.0',
        'matplotlib>=3.5.0',
        'pandas>=1.3.0',
        'Pillow>=8.0.0'
    ]
    
    print("🔧 Installing required packages...")
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ Installed: {package}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install: {package}")
            print("Please install manually using: pip install", package)

def create_project_structure():
    """Create the project folder structure"""
    folders = [
        'plant_images',
        'results',
        'visualizations',
        'sample_images',
        'documentation'
    ]
    
    print("\n📁 Creating project structure...")
    for folder in folders:
        Path(folder).mkdir(exist_ok=True)
        print(f"✅ Created folder: {folder}")

def create_sample_readme():
    """Create a comprehensive README file"""
    readme_content = """# 🌱 Plant Growth Tracking System

**Computer Graphics and Image Processing Project**

## 👥 Team Members
- **Grishma** - 4SF22CS071
- **Shreya Shenoy** - 4SF22CS206

## 👩‍🏫 Project Guide
**Mrs Srividya S**  
Assistant Professor

---

## 📋 Project Overview

This system uses OpenCV and computer vision techniques to track and analyze plant growth over time through image analysis. It provides quantitative metrics and visualizations to help users monitor their plant's health and development.

## ✨ Key Features

### 📊 Analysis Metrics
- **📏 Bounding Box Area**: Estimates plant size and growth coverage
- **🌿 Green Pixel Analysis**: Measures foliage density and spread
- **🍃 Leaf Count Estimation**: Basic count of distinct leaves
- **💚 Color Health Index**: Analyzes color distribution for health assessment
- **☀️ Brightness Analysis**: Estimates light conditions
- **📈 Growth Trend Visualization**: Charts showing progress over time
- **🔄 Delta Analysis**: Shows changes between consecutive uploads

### 🛠️ Technical Features
- **Image Preprocessing**: Noise reduction and normalization
- **Advanced Color Segmentation**: HSV-based green region detection
- **Morphological Operations**: Noise removal and shape enhancement
- **Contour Analysis**: Leaf counting and area calculation
- **Statistical Analysis**: Comprehensive growth metrics
- **Data Export**: JSON format for integration
- **Visualization Dashboard**: Multi-metric charts

---

## 🚀 Quick Start Guide

### 1. Installation
```bash
python setup.py
```

### 2. Add Your Plant Images
- Place images in the `plant_images/` folder
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`
- Recommended naming: `plant_2024-01-15.jpg` (includes date)

### 3. Run Analysis
```bash
python plant_growth_tracker.py
```

### 4. View Results
- **JSON Data**: `results/growth_analysis.json`
- **Visualizations**: `visualizations/growth_analysis.png`
- **Console Report**: Detailed analysis summary

---

## 📁 Project Structure

```
plant-growth-tracker/
├── plant_growth_tracker.py    # Main application
├── setup.py                   # Setup script
├── requirements.txt           # Dependencies
├── README.md                 # This file
├── plant_images/             # 📷 Place your plant photos here
├── results/                  # 📊 Analysis outputs (JSON)
├── visualizations/           # 📈 Generated charts
├── sample_images/            # 🌿 Sample data for testing
└── documentation/            # 📝 Project documentation
```

---

## 📷 Image Guidelines

### Best Practices
1. **📐 Consistent Framing**: Same angle and distance
2. **🌞 Good Lighting**: Natural, even lighting preferred
3. **🎯 Clear Background**: Minimal clutter behind plant
4. **📅 Regular Intervals**: Weekly or bi-weekly photos
5. **📝 Date Naming**: Include dates in filenames

### Naming Convention
- `plant_2024-01-15.jpg` (recommended)
- `my_plant_week1.jpg` (acceptable)
- Any format with clear chronological order

---

## 🔬 Technical Implementation

### Core Technologies
- **🐍 Python 3.7+**: Main programming language
- **👁️ OpenCV**: Computer vision and image processing
- **📊 NumPy**: Numerical computations
- **📈 Matplotlib**: Data visualization
- **📋 Pandas**: Data analysis and manipulation

### Image Processing Pipeline
1. **📥 Image Loading**: Multi-format support
2. **🔧 Preprocessing**: Resize, blur, noise reduction
3. **🎨 Color Segmentation**: HSV-based green detection
4. **🔍 Morphological Operations**: Shape cleaning
5. **📏 Feature Extraction**: Area, contours, pixels
6. **📊 Metric Calculation**: Growth indicators
7. **📈 Visualization**: Multi-metric dashboard

### Algorithm Details
- **Green Detection**: HSV color space thresholding
- **Leaf Counting**: Contour-based segmentation
- **Health Assessment**: Color saturation analysis
- **Growth Tracking**: Temporal metric comparison

---

## 📊 Output Explanation

### Growth Metrics
- **Size Growth**: Percentage change in plant area
- **Foliage Density**: Green pixel ratio over time
- **Leaf Development**: Estimated leaf count changes
- **Health Index**: Color-based health score (0-100)
- **Light Conditions**: Brightness analysis for care tips

### Visualization Dashboard
- **📈 6-Panel Chart**: Comprehensive growth overview
- **🔄 Delta Analysis**: Change detection between photos
- **📋 Summary Report**: Key insights and recommendations

---

## 🎯 Use Cases

### 🏠 Home Gardening
- Track houseplant development
- Monitor plant health remotely
- Optimize care routines

### 🌱 Plant Research
- Document growth experiments
- Compare treatment effects
- Generate research data

### 📚 Educational Projects
- Demonstrate computer vision concepts
- Teach image processing techniques
- Showcase real-world applications

---

## 🔧 Troubleshooting

### Common Issues

**❌ "No images found"**
- Check image file extensions (.jpg, .png, etc.)
- Ensure images are in `plant_images/` folder

**❌ "Could not read image"**
- Verify image file is not corrupted
- Check file permissions

**❌ Low accuracy results**
- Ensure consistent lighting conditions
- Use clear, uncluttered backgrounds
- Maintain same camera angle/distance

**❌ Installation problems**
- Update pip: `pip install --upgrade pip`
- Install packages individually
- Check Python version (3.7+ required)

---

## 📈 Future Enhancements

### Planned Features
- **🤖 ML Integration**: Advanced plant species detection
- **📱 Mobile App**: Real-time photo capture
- **☁️ Cloud Storage**: Online data synchronization
- **🌡️ Environmental Sensors**: Temperature/humidity integration
- **📧 Smart Alerts**: Automated care notifications

### Research Extensions
- **🔬 Disease Detection**: Early problem identification
- **🌿 Species Classification**: Automatic plant identification
- **📊 Predictive Analytics**: Growth forecasting
- **🌍 Community Features**: Data sharing and comparison

---

## 👨‍💻 Development Team

This project was developed as part of the Computer Graphics and Image Processing course, demonstrating practical applications of OpenCV and digital image analysis techniques.

### Contact Information
- **Technical Questions**: Contact team members
- **Academic Guidance**: Mrs Srividya S

---

## 📄 License & Usage

This project is developed for educational purposes as part of academic coursework. Feel free to use and modify for learning and research purposes.

---

*Happy Plant Growing! 🌱*
"""
    
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print("✅ Created comprehensive README.md")

def create_requirements_file():
    """Create requirements.txt file"""
    requirements_content = """# Plant Growth Tracking System - Requirements
# Computer Graphics and Image Processing Project

# Core Dependencies
opencv-python>=4.5.0
numpy>=1.21.0
matplotlib>=3.5.0
pandas>=1.3.0
Pillow>=8.0.0

# Optional for enhanced functionality
scipy>=1.7.0
scikit-image>=0.18.0
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements_content)
    
    print("✅ Created requirements.txt")

def create_sample_data():
    """Create sample images guide"""
    sample_guide = """# 🌿 Sample Images Guide

## Where to Get Plant Images

### 📷 Your Own Photos
1. Take photos of the same plant over time
2. Use consistent lighting and background
3. Maintain same camera angle and distance
4. Name files with dates: plant_2024-01-15.jpg

### 🌐 Sample Data Sources
For testing purposes, you can use:

1. **Free Stock Photos**:
   - Unsplash.com (search "plant growth", "seedling")
   - Pixabay.com (plant time-lapse images)
   - Pexels.com (plant development photos)

2. **Create Test Sequence**:
   - Find 3-5 images of plant at different growth stages
   - Rename them chronologically
   - Place in plant_images/ folder

### 📝 Naming Examples
```
plant_images/
├── plant_2024-01-01.jpg  (seedling)
├── plant_2024-01-15.jpg  (small sprout)
├── plant_2024-02-01.jpg  (growing)
├── plant_2024-02-15.jpg  (mature)
└── plant_2024-03-01.jpg  (full grown)
```

### ⚠️ Important Notes
- Minimum 2 images needed for analysis
- Use same plant species in sequence
- Maintain similar photo conditions
- Supported formats: .jpg, .jpeg, .png, .bmp
"""
    
    with open('sample_images/README.md', 'w') as f:
        f.write(sample_guide)
    
    print("✅ Created sample images guide")

def create_documentation():
    """Create technical documentation"""
    tech_doc = """# 🔬 Technical Documentation

## System Architecture

### Core Classes
- **PlantGrowthTracker**: Main analysis engine
  - Image preprocessing
  - Feature extraction
  - Metric calculation
  - Visualization generation

### Key Methods

#### Image Processing
```python
preprocess_image(image)          # Resize, blur, noise reduction
extract_green_mask(image)        # HSV-based plant segmentation
calculate_bounding_box_area()    # Size estimation
count_green_pixels()             # Foliage density
estimate_leaf_count()            # Contour-based counting
```

#### Analysis Functions
```python
analyze_brightness_contrast()    # Light condition assessment
calculate_color_health_index()   # Health score calculation
generate_delta_analysis()        # Change detection
create_growth_visualizations()   # Chart generation
```

## Algorithm Details

### Green Region Detection
1. Convert BGR → HSV color space
2. Define green color range: H(35-85), S(40-255), V(40-255)
3. Apply morphological operations (opening, closing)
4. Filter noise with kernel operations

### Leaf Counting Algorithm
1. Find external contours in green mask
2. Filter by minimum area threshold (100 pixels)
3. Count significant contours as leaves
4. Return estimated leaf count

### Health Index Calculation
1. Extract HSV values from green regions only
2. Calculate mean saturation and value
3. Weight: 70% saturation + 30% brightness
4. Normalize to 0-100 scale

### Growth Score Formula
```
Growth Score = (
    (bbox_area / max_bbox_area) × 0.4 +
    (green_ratio / max_green_ratio) × 0.3 +
    (leaf_count / max_leaf_count) × 0.2 +
    (health_index / 100) × 0.1
) × 100
```

## Performance Considerations

### Image Processing
- Images resized to max 800px for consistent processing
- Gaussian blur (5×5 kernel) for noise reduction
- Morphological operations with 5×5 kernel structure

### Memory Management
- Process images individually to minimize memory usage
- Results stored in lightweight dictionary format
- Automatic cleanup of intermediate processing data

## Error Handling

### Common Scenarios
- Invalid image files: Skip with error message
- Empty green masks: Return zero values
- Missing date information: Use file modification time
- No images found: Display helpful guidance

## Output Formats

### JSON Structure
```json
{
  "filename": "plant_2024-01-15.jpg",
  "date": "2024-01-15",
  "bbox_area": 15420,
  "green_pixels": 8934,
  "green_ratio": 0.1245,
  "leaf_count": 7,
  "brightness": 142.35,
  "contrast": 67.82,
  "health_index": 78.9,
  "bbox_coords": [45, 123, 234, 187]
}
```

### Visualization Components
- 6-panel dashboard with growth metrics
- Line plots with markers and grid
- Color-coded trend indicators
- High-resolution PNG output (300 DPI)

## Calibration Parameters

### Color Thresholds (HSV)
- Green Lower: [35, 40, 40]
- Green Upper: [85, 255, 255]
- Adjustable for different plant types

### Morphological Kernels
- Opening/Closing: 5×5 ones kernel
- Gaussian Blur: 5×5 kernel, σ=0 (auto)

### Area Thresholds
- Minimum leaf area: 100 pixels
- Image resize threshold: 800 pixels max dimension

## Extension Points

### Custom Metrics
Add new analysis functions following the pattern:
```python
def calculate_custom_metric(self, image, mask):
    # Your analysis code here
    return metric_value
```

### Additional Visualizations
Extend visualization function with new subplot:
```python
axes[row, col].plot(data)
axes[row, col].set_title('New Metric')
```

### Different Plant Types
Adjust HSV thresholds for specific species:
```python
# For flowering plants
lower_range = np.array([25, 30, 30])
upper_range = np.array([95, 255, 255])
```
"""
    
    with open('documentation/technical_guide.md', 'w') as f:
        f.write(tech_doc)
    
    print("✅ Created technical documentation")

def run_setup():
    """Run the complete setup process"""
    print("🌱 Plant Growth Tracking System - Setup")
    print("=" * 50)
    print("Authors: Shreya Shenoy (4SF22CS206), Grishma (4SF22CS071)")
    print("Guide: Mrs Srividya S")
    print("=" * 50)
    
    # Install requirements
    install_requirements()
    
    # Create project structure
    create_project_structure()
    
    # Create documentation
    create_sample_readme()
    create_requirements_file()
    create_sample_data()
    create_documentation()
    
    print("\n" + "✅" * 20)
    print("🎉 SETUP COMPLETE!")
    print("=" * 50)
    print("\n📋 NEXT STEPS:")
    print("1. 📷 Add plant images to 'plant_images/' folder")
    print("2. 🚀 Run: python plant_growth_tracker.py")
    print("3. 📊 Check results in 'results/' and 'visualizations/'")
    print("4. 📖 Read README.md for detailed instructions")
    print("\n💡 TIP: Start with 3-5 sample images for testing!")

if __name__ == "__main__":
    run_setup()