# Dataset Documentation

This project utilizes the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset for training and benchmarking traffic sign classification models.

## Dataset Overview

- **Classes**: 43 distinct traffic sign categories
- **Total Samples**: > 50,000 real-world images
- **Image Resolution**: Variable, from $15 \times 15$ to $250 \times 250$ pixels
- **Format**: RGB PPM / PNG images with associated CSV bounding box and track annotations

## Class Categories Breakdown

### 1. Speed Limit & Derestriction Signs (Classes 0-8, 32)
- 0: Speed limit (20km/h)
- 1: Speed limit (30km/h)
- 2: Speed limit (50km/h)
- 3: Speed limit (60km/h)
- 4: Speed limit (70km/h)
- 5: Speed limit (80km/h)
- 6: End of speed limit (80km/h)
- 7: Speed limit (100km/h)
- 8: Speed limit (120km/h)
- 32: End of all speed and passing limits

### 2. Prohibitory Signs (Classes 9, 10, 15, 16, 17, 41, 42)
- 9: No passing
- 10: No passing for vehicles over 3.5 metric tons
- 15: No vehicles
- 16: Vehicles over 3.5 metric tons prohibited
- 17: No entry
- 41: End of no passing
- 42: End of no passing by vehicles over 3.5 metric tons

### 3. Priority Signs (Classes 11, 12, 13, 14)
- 11: Right-of-way at next intersection
- 12: Priority road
- 13: Yield
- 14: Stop

### 4. Hazard & Warning Signs (Classes 18-31)
- 18: General caution
- 19: Dangerous curve to the left
- 20: Dangerous curve to the right
- 21: Double curve
- 22: Bumpy road
- 23: Slippery road
- 24: Road narrows on the right
- 25: Road work
- 26: Traffic signals
- 27: Pedestrians
- 28: Children crossing
- 29: Bicycles crossing
- 30: Beware of ice/snow
- 31: Wild animals crossing

### 5. Mandatory Direction Signs (Classes 33-40)
- 33: Turn right ahead
- 34: Turn left ahead
- 35: Ahead only
- 36: Go straight or right
- 37: Go straight or left
- 38: Keep right
- 39: Keep left
- 40: Roundabout mandatory

## Preprocessing & Data Augmentations

1. **Resolution Normalization**: Resized to uniform $48 \times 48$ dimension via bilinear/bicubic interpolation.
2. **Contrast & Brightness Adjustment**: Random color jittering ($\pm 20\%$) to simulate variable sunlight and overcast conditions.
3. **Geometric Augmentations**: Random small-angle rotations ($\pm 15^\circ$) and scaling to account for camera angles.
4. **Standardization**: Channel-wise z-score normalization using standard ImageNet parameters ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
