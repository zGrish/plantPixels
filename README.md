# 🌱 Plant Growth Tracking System

**Computer Graphics and Image Processing Project**

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
