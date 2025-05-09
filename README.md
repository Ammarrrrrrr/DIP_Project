# Handwritten Text Recognition with Llama 3.2-Vision

This project implements an advanced OCR (Optical Character Recognition) system specifically designed for handwritten text recognition using Llama 3.2-Vision. The system combines state-of-the-art image processing techniques with powerful AI capabilities to accurately transcribe handwritten text.

## Features

- Advanced image preprocessing pipeline
- Support for various handwriting styles
- Real-time performance metrics
- User-friendly Streamlit interface
- Comprehensive error handling
- Image size optimization

## Technical Details

### Image Processing Pipeline

1. **Preprocessing Steps**:
   - Grayscale conversion
   - Adaptive thresholding
   - Denoising
   - Contrast enhancement

2. **Performance Optimization**:
   - Automatic image size validation
   - Efficient OpenCV operations
   - Processing time tracking

3. **Recognition Features**:
   - Handles cursive and print handwriting
   - Preserves line structure
   - Context-aware character recognition
   - Intelligent word reconstruction

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/llama-ocr.git
cd llama-ocr
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure Llama 3.2-Vision is running locally on port 11434

## Usage

1. Start the Streamlit app:
```bash
streamlit run ocr.py
```

2. Upload an image containing handwritten text
3. View the preprocessing results
4. Click "Run Handwritten Text Recognition" to process the image

## Performance Metrics

The system provides the following metrics:
- Processing time
- Image dimensions
- Character-level accuracy (when ground truth is available)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Llama 3.2-Vision team for the powerful vision model
- OpenCV community for image processing tools
- Streamlit for the web interface framework

# Code Explanation for OCR Assistant with Llama 3.2-Vision

Following provides a line-by-line explanation of the Python code used for building the OCR assistant using Streamlit, Llama 3.2-Vision, and Ollama.


```python
import streamlit as st
import base64
import requests
from PIL import Image
import os
import json
```
- **Import Libraries**: Import required Python libraries, including Streamlit for the UI, `base64` for encoding images, `requests` for making HTTP requests, `PIL` for image handling, `os` for file operations, and `json` for JSON data.

```python
SYSTEM_PROMPT = """You are an advanced OCR tool. Your task is to accurately transcribe the text from the provided image...
"""
```
- **System Prompt**: Defines the guidelines for transcription to be followed by the OCR tool.

### Encode Image to Base64
```python
def encode_image_to_base64(image_path):
    """Convert an image file to a base64 encoded string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
```
- **encode_image_to_base64(image_path)**: This function takes an image path and converts the image to a base64-encoded string.

### Parse Response
```python
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

        return "\n".join(combined_result)

    except Exception as e:
        st.error(f"Error parsing response: {str(e)}")
        return response_text
```
- **parse_response(response_text)**: Parses the response text from the OCR model and handles exceptions if parsing fails.

### Perform OCR
```python
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
```
- **perform_ocr(image_path)**: This function sends the encoded image to an OCR endpoint and returns the parsed result.

### Main Function
```python
def main():
    try:
        logo_base64 = base64.b64encode(open('MinimalDevopsLogo.png', 'rb').read()).decode('utf-8')
        st.markdown(
            """
            <div style="display: flex; align-items: center;">
                <img src="data:image/png;base64,{}" width="50" style="margin-right: 10px;"/>
                <h1>OCR Assistant with Llama 3.2-Vision</h1>
            </div>
            """.format(logo_base64), unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.title("OCR Assistant with Llama 3.2-Vision")

    uploaded_file = st.file_uploader("Upload an image file for OCR analysis", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        os.makedirs("temp", exist_ok=True)
        
        with open(os.path.join("temp", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
            image_path = f.name
        
        image = Image.open(image_path)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if st.button("Run OCR"):
            initial_result = perform_ocr(image_path)
            if initial_result:
                st.subheader("OCR Recognition Result:")
                st.text(initial_result.replace("\n", " "))
```
- **main()**: The main function defines the Streamlit UI. It handles image upload, displays the image, and runs the OCR when the user clicks the button.

```python
if __name__ == "__main__":
    main()
```
- **if __name__ == "__main__": main()**: Runs the `main()` function when the script is executed directly.

## How to Use
1. Install the required libraries:
   ```sh
   pip install streamlit requests pillow
   ```
2. Run the application using Streamlit:
   ```sh
   streamlit run your_script.py
   ```
3. Upload an image, and click on **Run OCR** to perform the OCR analysis.

## Requirements
- Python 3.x
- Streamlit
- Requests
- Pillow (PIL)


This explanation should give you a clear understanding of how each part of the code contributes to the overall OCR assistant functionality. Feel free to experiment by adding more features or making customizations to the code!

