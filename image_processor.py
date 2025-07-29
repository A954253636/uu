import cv2
import numpy as np
import os
import argparse
from pathlib import Path

class ImageProcessor:
    def __init__(self):
        pass
    
    def analyze_brightness(self, image):
        """
        Analyze the brightness of an image
        Returns a value between 0 (dark) and 1 (bright)
        """
        # For color images, convert to HSV and use the V channel
        if len(image.shape) == 3:
            # Convert BGR to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            # Extract the V channel (brightness)
            v_channel = hsv[:, :, 2]
            # Calculate average brightness (normalized to 0-1)
            return np.mean(v_channel) / 255.0
        else:
            # For grayscale images, directly calculate mean brightness
            return np.mean(image) / 255.0
    
    def get_adaptive_brightness_adjustment(self, brightness_level):
        """
        Determine brightness adjustment based on current image brightness
        Args:
            brightness_level: Value between 0 and 1 representing current brightness
        Returns:
            Percentage adjustment to apply
        """
        # Dark images (brightness_level < 0.3): apply large adjustment (+100 to +120)
        if brightness_level < 0.3:
            # The darker the image, the stronger the adjustment
            return 50 + (0.3 - brightness_level) * 200
            
        # Medium brightness (0.3 <= brightness_level < 0.6): apply moderate adjustment
        elif brightness_level < 0.6:
            return 15 + (0.6 - brightness_level) * 80
            
        # Bright images (brightness_level >= 0.6): apply minimal adjustment
        else:
            # For very bright images, possibly slight negative adjustment
            return 5 - (brightness_level - 0.6) * 40
    
    def adjust_brightness(self, image, percentage):
        """Adjust brightness by percentage (-100 to 100)"""
        factor = 1 + percentage / 100
        return np.clip(image * factor, 0, 255).astype(np.uint8)
        
    def adjust_contrast(self, image, percentage):
        """Adjust contrast by percentage (-100 to 100) using a curve-based approach
        This provides a more natural contrast adjustment similar to photo editing software"""
        # Convert percentage to a reasonable factor (0 to 3)
        # This maps percentage range -100 to 100 to factor range 0 to 3
        factor = max(0, (percentage + 100) / 100)
        
        # Convert to float for processing
        img_float = image.astype(np.float32) / 255.0
        
        # Apply contrast curve (using a sigmoid-like function)
        # This creates an S-curve that preserves more details in shadows and highlights
        if percentage > 0:
            # Positive contrast: apply S-curve
            img_float = (1.0 / (1.0 + np.exp(-(img_float - 0.5) * factor * 3))) 
            # The *3 adjusts the steepness to make the effect more visible at lower percentages
        else:
            # Negative contrast: apply inverse S-curve
            img_float = 0.5 + (img_float - 0.5) * factor
        
        # Convert back to 0-255 range
        return np.clip(img_float * 255, 0, 255).astype(np.uint8)
    
    def enhance_shadows(self, image, percentage):
        """Enhance shadow brightness by percentage (0 to 100)
        This increases the brightness of dark areas while preserving highlights"""
        # Convert to float for processing
        img_float = image.astype(np.float32)
        
        # Define shadow threshold (pixels below this are considered shadows)
        threshold = 100
        
        # Create a weight mask that's 1 for complete shadows and 0 for bright areas
        shadow_mask = np.clip((threshold - img_float) / threshold, 0, 1)
        
        # Calculate brightening amount (50% means adding up to 50% of max value in darkest areas)
        brightening = shadow_mask * (255 * percentage / 100)
        
        # Apply brightening to the image
        result = np.clip(img_float + brightening, 0, 255).astype(np.uint8)
        return result
    
    def enhance_structure(self, image, percentage):
        """Enhance structure by percentage (0 to 100+)"""
        # Convert to float for processing
        img_float = image.astype(np.float32) / 255.0
        
        # Extract structure layer using bilateral filter
        sigma_color, sigma_space = 10, 10
        structure = cv2.bilateralFilter(img_float, -1, sigma_color, sigma_space)
        
        # Extract detail layer
        detail = img_float - structure
        
        # Enhance detail layer based on percentage
        strength = 1 + percentage / 100
        enhanced = structure + detail * strength
        
        # Clip values and convert back
        result = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
        return result
    
    def process_image(self, image, brightness=None, contrast=30, shadows=50, structure=100):
        """Apply all adjustments to image with configurable parameters"""
        # Make a copy to avoid modifying the original
        result = image.copy()
        
        # Analyze image brightness and determine adjustment if not specified
        if brightness is None:
            img_brightness = self.analyze_brightness(image)
            brightness = self.get_adaptive_brightness_adjustment(img_brightness)
            print(f"Image brightness level: {img_brightness:.2f}, applying adjustment: {brightness:.1f}%")
        
        # For color images, process each channel
        if len(image.shape) == 3:
            for i in range(3):  # Process each channel
                # Apply adjustments in sequence
                result[:,:,i] = self.enhance_shadows(result[:,:,i], shadows)
                
                result[:,:,i] = self.adjust_brightness(result[:,:,i], brightness)
                
                result[:,:,i] = self.adjust_contrast(result[:,:,i], contrast)
                
                result[:,:,i] = self.enhance_structure(result[:,:,i], structure)
        else:
            # Grayscale image
            result = self.adjust_brightness(result, brightness)
            result = self.adjust_contrast(result, contrast)
            result = self.enhance_shadows(result, shadows)
            result = self.enhance_structure(result, structure)
            
        return result

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Process images with specified adjustments')
    parser.add_argument('input_dir', help='Directory containing input images')
    parser.add_argument('output_dir', help='Directory to save processed images')
    parser.add_argument('--brightness', type=float, default=None, help='Brightness adjustment (-100 to 100) or None for automatic')
    parser.add_argument('--contrast', type=float, default=95, help='Contrast adjustment (-100 to 100)')
    parser.add_argument('--shadows', type=float, default=20, help='Shadow enhancement (0 to 100)')
    parser.add_argument('--structure', type=float, default=25, help='Structure enhancement (0 to 100+)')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize processor
    processor = ImageProcessor()
    
    # Supported image extensions
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    # Process each image in the input directory
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    
    image_files = [f for f in input_path.iterdir() if f.suffix.lower() in extensions]
    
    print(f"Found {len(image_files)} images to process.")
    
    for img_path in image_files:
        try:
            print(f"Processing {img_path.name}...")
            
            # Read image
            image = cv2.imread(str(img_path))
            
            if image is None:
                print(f"Error reading {img_path.name}, skipping.")
                continue
                
            # Process image
            processed = processor.process_image(
                image, 
                brightness=args.brightness,
                contrast=args.contrast,
                shadows=args.shadows,
                structure=args.structure
            )
            
            # Save processed image
            output_file = output_path / img_path.name
            cv2.imwrite(str(output_file), processed)
            
            print(f"Saved to {output_file.name}")
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {str(e)}")
    
    print("Processing complete!")

if __name__ == "__main__":
    main()