# Handwritten Text Recognition System

An advanced handwritten text recognition system that combines traditional image processing techniques with state-of-the-art AI models. The system achieves high accuracy in recognizing various handwriting styles while maintaining computational efficiency.

![Project Logo](MinimalDevopsLogo.png)

## 🌟 Features

- **Advanced Image Processing**
  - Adaptive thresholding
  - Noise reduction
  - Contrast enhancement
  - Real-time preprocessing

- **AI-Powered Recognition**
  - Llama 3.2-Vision model integration
  - High accuracy text recognition
  - Support for various handwriting styles
  - Context-aware processing

- **User-Friendly Interface**
  - Streamlit web application
  - Real-time processing
  - Visual feedback
  - Performance metrics

## 📋 Prerequisites

- Python 3.8 or higher
- Ollama (for AI model)
- Git (for cloning the repository)

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/handwritten-text-recognition.git
cd handwritten-text-recognition
```

2. Create and activate virtual environment:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Install Ollama:
- Windows: Download from [Ollama Website](https://ollama.ai)
- Linux/Mac: Follow [Ollama Installation Guide](https://ollama.ai/docs/installation)

5. Pull the required model:
```bash
ollama pull llama3.2-vision
```

## 💻 Usage

1. Start the application:
```bash
# On Windows
start_full.bat

# On Linux/Mac
./start_full.sh
```

2. Open your web browser and navigate to:
```
http://localhost:8501
```

3. Upload an image containing handwritten text

4. View the results:
   - Original image
   - Preprocessed stages
   - Recognized text
   - Performance metrics

## 🛠️ Project Structure

```
handwritten-text-recognition/
├── ocr.py                 # Main application code
├── requirements.txt       # Python dependencies
├── start_full.bat        # Windows startup script
├── start_full.sh         # Linux/Mac startup script
├── project_report.tex    # Project documentation
└── README.md             # This file
```

## 📊 Performance

- Accuracy: 92% on test dataset
- Processing Time: < 2 seconds per image
- Memory Usage: < 500MB
- CPU Utilization: < 30%
- GPU Memory: < 1GB

## 🔧 Technical Details

### Image Processing Pipeline
1. Grayscale Conversion
2. Adaptive Thresholding
3. Noise Reduction
4. Contrast Enhancement
5. AI Model Processing

### Technologies Used
- Python 3.8+
- Streamlit
- NumPy
- OpenCV
- Llama 3.2-Vision
- scikit-learn

## 📝 Project Report

A detailed project report is available in `project_report.tex`. It includes:
- System architecture
- Implementation details
- Performance analysis
- Future improvements

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- Ammar Elsayed (222321)
- Eslam Ahmed (220921)
- Mohamed Ashraf (213473)

## 🙏 Acknowledgments

- Dr. Ahmed Ayoub (Instructor)
- MSA University
- Department of Computer Engineering

## 📞 Support

For support, please open an issue in the GitHub repository or contact the authors.

## 🔄 Updates

- Latest update: March 2024
- Version: 1.0.0

---

Made with ❤️ by MSA University Students
