import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
print("OpenCV Version:", cv2.__version__)
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def analyze_cotton_leaf(image_path):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        return None, "Error: Could not load image"
    
    # Create a copy for display
    display_img = img.copy()
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define green color range for leaf segmentation
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([90, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Refine leaf mask with morphological operations
    kernel = np.ones((7, 7), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    
    # Find the largest contour (the leaf)
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, "Error: No leaf detected"
        
    leaf_contour = max(contours, key=cv2.contourArea)
    leaf_area = cv2.contourArea(leaf_contour)
    if leaf_area <= 0:
        return None, "Invalid leaf area detected"
    if leaf_area < 500:  # Minimum leaf area threshold
        return None, "Error: Leaf too small or not detected"
    
    # Create a mask for the leaf area
    leaf_mask = np.zeros_like(green_mask)
    cv2.drawContours(leaf_mask, [leaf_contour], -1, 255, -1)
    
    # Extract only the leaf region
    leaf_only = cv2.bitwise_and(img, img, mask=leaf_mask)
    
    # Convert leaf region to HSV for analysis
    leaf_hsv = cv2.cvtColor(leaf_only, cv2.COLOR_BGR2HSV)
    
    # Define healthy green color range
    lower_healthy = np.array([35, 40, 40])
    upper_healthy = np.array([85, 255, 255])
    healthy_mask = cv2.inRange(leaf_hsv, lower_healthy, upper_healthy)
    
    # Calculate healthy area
    healthy_pixels = cv2.countNonZero(healthy_mask)
    healthy_percentage = (healthy_pixels / leaf_area) * 100
    
    # Detect damaged areas (non-green within leaf)
    damaged_mask = cv2.bitwise_and(leaf_mask, cv2.bitwise_not(healthy_mask))
    
    # Find damaged contours
    damage_contours, _ = cv2.findContours(damaged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter small contours (noise)
    min_damage_area = leaf_area * 0.002  # 0.2% of leaf area
    significant_damage = [cnt for cnt in damage_contours if cv2.contourArea(cnt) > min_damage_area]
    
    # Calculate damaged area
    damaged_area = sum(cv2.contourArea(cnt) for cnt in significant_damage)
    damaged_percentage = (damaged_area / leaf_area) * 100
    
    # Detect specific disease indicators
    disease_detected = False
    disease_reasons = []
    
    # 1. Check for yellowing
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    yellow_mask = cv2.inRange(leaf_hsv, lower_yellow, upper_yellow)
    yellow_area = cv2.countNonZero(yellow_mask) / leaf_area * 100
    if yellow_area > 5:
        disease_detected = True
        disease_reasons.append(f"Yellowing detected ({yellow_area:.1f}%)")
    
    # 2. Check for brown spots
    lower_brown = np.array([10, 100, 20])
    upper_brown = np.array([20, 255, 200])
    brown_mask = cv2.inRange(leaf_hsv, lower_brown, upper_brown)
    brown_area = cv2.countNonZero(brown_mask) / leaf_area * 100
    if brown_area > 2:
        disease_detected = True
        disease_reasons.append(f"Brown spots detected ({brown_area:.1f}%)")
    
    # 3. Check for holes (insect damage)
    hole_contours = []
    for cnt in significant_damage:
        # Check for contour that is fully surrounded by leaf
        x, y, w, h = cv2.boundingRect(cnt)
        if x > 5 and y > 5 and x+w < img.shape[1]-5 and y+h < img.shape[0]-5:
            # Check if it's a hole (darker center than edges)
            center_region = leaf_only[y:y+h, x:x+w]
            if center_region.size == 0:  # Skip if empty
                continue
            center_color = np.mean(center_region)
            edge_region = leaf_only[max(0, y-3):y, max(0, x):min(x+w, img.shape[1])]
            if edge_region.size == 0:  # Skip if empty
                continue
            edge_color = np.mean(edge_region)
            if center_color < edge_color * 0.8:  # Center is darker
                hole_contours.append(cnt)
    
    if hole_contours:
        disease_detected = True
        disease_reasons.append(f"{len(hole_contours)} possible insect holes")
    
    # 4. Check overall health thresholds
    if healthy_percentage < 95:
        disease_detected = True
        disease_reasons.append(f"Low healthy area ({healthy_percentage:.1f}%)")
    
    if damaged_percentage > 3:
        disease_detected = True
        disease_reasons.append(f"Significant damage ({damaged_percentage:.1f}%)")
    
    if len(significant_damage) > 5:
        disease_detected = True
        disease_reasons.append(f"Multiple damaged spots ({len(significant_damage)})")
    
    # Visualization
    # Draw leaf boundary
    cv2.drawContours(display_img, [leaf_contour], -1, (0, 255, 0), 3)
    
    # Draw damaged areas
    cv2.drawContours(display_img, significant_damage, -1, (0, 0, 255), 2)
    
    # Highlight holes
    cv2.drawContours(display_img, hole_contours, -1, (255, 0, 0), 3)
    
    # Add text overlay
    status = "DISEASED" if disease_detected else "HEALTHY"
    color = (0, 0, 255) if disease_detected else (0, 255, 0)
    cv2.putText(display_img, f"Status: {status}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(display_img, f"Healthy: {healthy_percentage:.1f}%", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display_img, f"Damaged: {damaged_percentage:.1f}%", (20, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Add disease reasons
    for i, reason in enumerate(disease_reasons[:3]):  # Show up to 3 reasons
        cv2.putText(display_img, reason, (20, 160 + i*30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    
    # Save annotated image - use relative path
    original_filename = os.path.basename(image_path)
    result_filename = "result_" + original_filename
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
    cv2.imwrite(result_path, display_img)
    
    # Prepare results with relative paths
    results = {
        "status": status,
        "healthy_percentage": f"{healthy_percentage:.1f}%",
        "damaged_percentage": f"{damaged_percentage:.1f}%",
        "damage_spots": len(significant_damage),
        "disease_reasons": disease_reasons,
        "result_image": f"uploads/{result_filename}",  # Relative path
    }
    
    return results, None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return render_template('index.html', error="No file selected")
        
        file = request.files['file']
        
        # If user does not select file
        if file.filename == '':
            return render_template('index.html', error="No file selected")
        
        if file:
            # Save the uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            print("Saved to:", filepath)

            img = cv2.imread(filepath)
            if img is None:
                return render_template(
                    "index.html",
                    error="OpenCV could not read the uploaded image."
                )
            
            # Get relative path for original image
            original_image_rel = f"uploads/{filename}"
            
            # Analyze the leaf
            try:
                results, error = analyze_cotton_leaf(filepath)

                if error:
                    return render_template(
                        "index.html",
                        error=error,
                        original_image=original_image_rel
                    )

                return render_template(
                    "index.html",
                    results=results,
                    original_image=original_image_rel
                )

            except Exception as e:
                import traceback
                traceback.print_exc()

                return render_template(
                    "index.html",
                    error=f"Server Error: {str(e)}"
                )

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
