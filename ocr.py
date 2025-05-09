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
    # Convert PIL Image to OpenCV format
    img = np.array(image)
    
    # Convert to grayscale if image is colored
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Apply adaptive thresholding
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Denoise the image
    img = cv2.fastNlMeansDenoising(img)
    
    # Enhance contrast
    img = cv2.equalizeHist(img)
    
    # Convert back to PIL Image
    return Image.fromarray(img)

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
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

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
    base64_image = encode_image_to_base64(image_path)
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "llama3.2-vision",
            "messages": [
                {
                    "role": "user",
                    "content": SYSTEM_PROMPT,
                    "images": [base64_image],
                },
            ],
        }
    )
    if response.status_code == 200:
        return parse_response(response.text)
    else:
        st.error(f"Error: {response.status_code} - {response.text}")
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
        
        with open(os.path.join("temp", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
            image_path = f.name
        
        # Load and validate image size
        original_image = Image.open(image_path)
        original_image = validate_image_size(original_image)
        processed_image = preprocess_image(original_image)
        
        # Display both original and processed images
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_image, caption="Original Image")
        with col2:
            st.image(processed_image, caption="Preprocessed Image")
        
        if st.button("Run Handwritten Text Recognition"):
            start_time = time.time()
            
            # Save processed image temporarily
            processed_path = os.path.join("temp", "processed_" + uploaded_file.name)
            processed_image.save(processed_path)
            
            initial_result = perform_ocr(processed_path)
            processing_time = time.time() - start_time
            
            if initial_result:
                st.subheader("Recognition Result:")
                st.text(initial_result.replace("\n", " "))
                
                # Display performance metrics
                st.subheader("Performance Metrics:")
                metrics = {
                    'Processing Time': f"{processing_time:.2f} seconds",
                    'Image Size': f"{original_image.size[0]}x{original_image.size[1]} pixels"
                }
                st.json(metrics)
            
            # Clean up temporary files
            try:
                os.remove(processed_path)
            except:
                pass

if __name__ == "__main__":
    main()
