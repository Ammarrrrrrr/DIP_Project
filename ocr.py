import streamlit as st
import base64
import requests
from PIL import Image
import os
import json
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
import time

def preprocess_image(image):
    """Preprocess the image for better handwritten text recognition."""
    # Convert PIL Image to numpy array
    img = np.array(image)
    
    # Convert to grayscale if image is colored - using vectorized operations
    if len(img.shape) == 3:
        img = np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    
    # Optimized adaptive thresholding using vectorized operations
    def manual_adaptive_threshold(img, block_size=11, C=2):
        height, width = img.shape
        result = np.zeros_like(img)
        
        # Use numpy's stride tricks for efficient block processing
        from numpy.lib.stride_tricks import sliding_window_view
        blocks = sliding_window_view(img, (block_size, block_size))
        
        # Calculate means for all blocks at once
        means = np.mean(blocks, axis=(2, 3))
        
        # Apply threshold using vectorized operations
        result[block_size//2:-(block_size//2), block_size//2:-(block_size//2)] = \
            (img[block_size//2:-(block_size//2), block_size//2:-(block_size//2)] > (means - C)) * 255
        
        return result
    
    # Optimized median filter using scipy's implementation
    from scipy.ndimage import median_filter
    def manual_median_filter(img, kernel_size=3):
        return median_filter(img, size=kernel_size)
    
    # Optimized histogram equalization using numpy operations
    def manual_histogram_equalization(img):
        # Calculate histogram using numpy's bincount
        hist = np.bincount(img.ravel(), minlength=256)
        
        # Calculate cumulative distribution function
        cdf = np.cumsum(hist)
        cdf_normalized = cdf * 255 / cdf[-1]
        
        # Apply equalization using numpy's vectorized operations
        return cdf_normalized[img].astype(np.uint8)
    
    # Apply preprocessing steps and store intermediate results
    with st.spinner('Converting to grayscale...'):
        grayscale = img.copy()
    
    with st.spinner('Applying adaptive thresholding...'):
        thresholded = manual_adaptive_threshold(grayscale)
    
    with st.spinner('Applying denoising...'):
        denoised = manual_median_filter(thresholded)
    
    with st.spinner('Enhancing contrast...'):
        enhanced = manual_histogram_equalization(denoised)
    
    # Return all stages for visualization
    return {
        'original': Image.fromarray(img),
        'grayscale': Image.fromarray(grayscale),
        'thresholded': Image.fromarray(thresholded),
        'denoised': Image.fromarray(denoised),
        'enhanced': Image.fromarray(enhanced)
    }

SYSTEM_PROMPT = """You are a highly advanced OCR system specialized in handwritten text recognition. Your task is to accurately transcribe all visible handwritten text from the provided image. 
Follow these detailed guidelines strictly to ensure precise and natural transcription:

1. **Handwriting Recognition Focus**:
   - Pay special attention to handwritten characters and their variations
   - Consider different handwriting styles and personal writing characteristics
   - Handle connected letters and cursive writing appropriately

2. **Preserve Original Line Structure**: 
   Maintain the exact line breaks and formatting as seen in the image. Each line in the transcription should correspond directly to a line in the image.

3. **Handle Handwriting Variations**:
   - Recognize different styles of handwriting (cursive, print, mixed)
   - Account for variations in letter formation
   - Handle overlapping or touching characters appropriately

4. **Reconstruct Complete Words**: 
   If a word is split across lines or unclear, intelligently reconstruct it as a complete, correctly spelled word.

5. **Make Sense of Unclear Characters**: 
   If a character appears distorted or unclear:
   - Use contextual clues from surrounding text
   - Consider common handwriting patterns
   - Make educated guesses based on word context
   - If no reasonable interpretation can be made, leave the character as-is

6. **No Additional Comments or Annotations**: 
   Output only the transcribed text. Do not include explanations, metadata, or any other form of commentary.

7. **Output Format**: 
   Return the transcription as a single block of text. Preserve all line breaks and ensure natural word spacing."""

def encode_image_to_base64(image_path):
    """Convert an image file to a base64 encoded string."""
    try:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            base64_data = base64.b64encode(image_data).decode('utf-8')
            print(f"Successfully encoded image to base64")
            return base64_data
    except Exception as e:
        print(f"Error encoding image: {str(e)}")
        raise

def parse_response(response_text):
    """Parse the response text from the model."""
    try:
        json_objects = response_text.splitlines()
        combined_result = []

        for json_object in json_objects:
            try:
                parsed_json = json.loads(json_object)
                content = parsed_json.get("message", {}).get("content", "")
                if content and content.strip():
                    combined_result.append(content.strip())
            except json.JSONDecodeError:
                combined_result.append(json_object.strip())

        # Join lines while preserving formatting
        return "\n".join(combined_result)

    except Exception as e:
        st.error(f"Error parsing response: {str(e)}")
        return response_text  # Return raw text if parsing fails

def perform_ocr(image_path):
    """Perform OCR on the given image using Llama 3.2-Vision."""
    try:
        # Debug: Print image path
        print(f"Processing image: {image_path}")
        
        # Check if image exists
        if not os.path.exists(image_path):
            st.error(f"Image file not found: {image_path}")
            return None
            
        # Debug: Print image size
        image_size = os.path.getsize(image_path)
        print(f"Image size: {image_size} bytes")
        
        # Encode image
        with st.spinner('Encoding image...'):
            base64_image = encode_image_to_base64(image_path)
            print(f"Base64 image length: {len(base64_image)}")
        
        # Prepare request
        request_data = {
            "model": "llama3.2-vision",
            "messages": [
                {
                    "role": "user",
                    "content": SYSTEM_PROMPT,
                    "images": [base64_image],
                },
            ],
        }
        
        # Debug: Print request URL
        print("Sending request to: http://localhost:11434/api/chat")
        
        # Send request with progress
        with st.spinner('Sending request to AI model...'):
            response = requests.post(
                "http://localhost:11434/api/chat",
                json=request_data,
                timeout=30  # Add timeout
            )
        
        # Debug: Print response status
        print(f"Response status code: {response.status_code}")
        
        if response.status_code == 200:
            with st.spinner('Processing AI response...'):
                result = parse_response(response.text)
                print(f"Parsed result: {result[:100]}...")  # Print first 100 chars
                return result
        else:
            error_msg = f"Error: {response.status_code} - {response.text}"
            print(error_msg)
            st.error(error_msg)
            return None
            
    except Exception as e:
        error_msg = f"Error during OCR processing: {str(e)}"
        print(error_msg)
        st.error(error_msg)
        return None

def calculate_metrics(predicted_text, ground_truth):
    """Calculate performance metrics for OCR results."""
    # Convert texts to lists of characters for comparison
    pred_chars = list(predicted_text.lower())
    true_chars = list(ground_truth.lower())
    
    # Pad shorter sequence with spaces
    max_len = max(len(pred_chars), len(true_chars))
    pred_chars.extend([' '] * (max_len - len(pred_chars)))
    true_chars.extend([' '] * (max_len - len(true_chars)))
    
    # Calculate metrics
    accuracy = accuracy_score(true_chars, pred_chars)
    precision = precision_score(true_chars, pred_chars, average='weighted', zero_division=0)
    recall = recall_score(true_chars, pred_chars, average='weighted', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    }

def validate_image_size(image, max_size=2000):
    """Validate and resize image if too large."""
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = tuple(int(dim * ratio) for dim in image.size)
        return image.resize(new_size, Image.LANCZOS)
    return image

def main():
    try:
        logo_base64 = base64.b64encode(open('MinimalDevopsLogo.png', 'rb').read()).decode('utf-8')
        st.markdown(
            """
            <div style="display: flex; align-items: center;">
                <img src="data:image/png;base64,{}" width="50" style="margin-right: 10px;"/>
                <h1>Handwritten Text Recognition with Llama 3.2-Vision</h1>
            </div>
            """.format(logo_base64), unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.title("Handwritten Text Recognition with Llama 3.2-Vision")
    
    # Add FAQ section
    with st.expander("Frequently Asked Questions"):
        st.markdown("""
        ### How does the OCR system work?
        The system uses advanced image processing techniques and Llama 3.2-Vision to recognize handwritten text. 
        It preprocesses images to enhance text visibility and then uses AI to transcribe the content.
        
        ### What types of handwriting can it recognize?
        The system can handle various handwriting styles including cursive, print, and mixed styles.
        
        ### How accurate is the system?
        Accuracy depends on image quality and handwriting clarity. The system shows performance metrics for each recognition.
        """)
    
    uploaded_file = st.file_uploader("Upload an image containing handwritten text", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        os.makedirs("temp", exist_ok=True)
        
        # Save uploaded file
        temp_path = os.path.join("temp", uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            image_path = f.name
        
        # Load and validate image size
        with st.spinner('Loading and validating image...'):
            original_image = Image.open(image_path)
            original_image = validate_image_size(original_image)
        
        # Get all preprocessing stages
        processed_stages = preprocess_image(original_image)
        
        # Display all preprocessing stages
        st.subheader("Image Preprocessing Stages")
        cols = st.columns(5)
        stages = ['original', 'grayscale', 'thresholded', 'denoised', 'enhanced']
        for col, stage in zip(cols, stages):
            with col:
                st.image(processed_stages[stage], caption=stage.capitalize())
        
        # Save processed image temporarily
        processed_path = os.path.join("temp", "processed_" + uploaded_file.name)
        processed_stages['enhanced'].save(processed_path)
        
        # Process with AI
        initial_result = perform_ocr(processed_path)
            
        if initial_result:
            st.subheader("Recognition Result:")
            st.text(initial_result.replace("\n", " "))
            
            # Display performance metrics
            st.subheader("Performance Metrics:")
            metrics = {
                'Image Size': f"{original_image.size[0]}x{original_image.size[1]} pixels",
                'File Size': f"{os.path.getsize(processed_path) / 1024:.2f} KB"
            }
            st.json(metrics)
        
        # Clean up temporary files
        try:
            os.remove(processed_path)
        except:
            pass

if __name__ == "__main__":
    main()
