# 🔬 Technical Documentation

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
