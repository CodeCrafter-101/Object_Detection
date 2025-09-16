# Object Detection

## 📖 Introduction: What is Object Detection?  

Object detection is a computer vision task that allows a model to:  
1. **Locate** objects in an image or video (bounding boxes)  
2. **Classify** the type of object (e.g., person, car, dog)  
3. **Measure confidence** in predictions (accuracy ratio / probability)  

Unlike simple image classification, object detection identifies **multiple objects** in the same frame and assigns labels to each one.  

YOLO is one of the most popular models for this because it is:  
- **Fast** → Processes images in real time  
- **Accurate** → High confidence scores  
- **Versatile** → Works with images, videos, and live webcam streams  

---

## 🛠️ Libraries & Tools Used  

- **YOLO** – Core object detection model  
- **Python** – Programming language  
- **OpenCV** – Video processing & webcam input  
- **Pillow** – Image processing and manipulation  
- **Streamlit** – Web app deployment framework  
- **Ngrok** – Tunnel service for deploying Streamlit apps with tokens  
- **Jupyter Notebook** – Development environment  

---

## 📂 Project Structure  
<img width="459" height="350" alt="image" src="https://github.com/user-attachments/assets/de19f2e0-a3f1-426e-ad59-6a2c51264247" />



---

## 🖼️ Object Detection on Images  

**Goal:** Detect objects in static images and display bounding boxes with confidence scores.  

**Steps Taken:**  
- ```
  !pip install -q ultralytics opencv-python-headless matplotlib pillow
  ```

- Import necessary libraries
  ```
  from ultralytics import YOLO
  import cv2
  from matplotlib import pyplot as plt
  from PIL import Image
  from google.colab import files
  ```
  
- Loaded sample images  
- Used YOLO pre-trained model for inference
  ```
  model = YOLO('yolov8n.pt')
  ```
  
- Processed results and displayed bounding boxes with labels  

**Key Libraries:**  
- Pillow  
- YOLO  

---

## 🎥 Object Detection on Videos  

**Goal:** Apply YOLO on videos frame-by-frame.  

**Steps Taken:**  
- Loaded a video file with OpenCV  
- Processed each frame through YOLO model  
- Saved the output video with bounding boxes  

**Key Libraries:**  
- OpenCV  
- YOLO  

---

## 📡 Real-Time Webcam Detection  

**Goal:** Run YOLO detections on a live webcam feed.  

**Steps Taken:**  
- ```
  !pip install -q ultralytics opencv-python-headless matplotlib pillow
  ```
- Import necerrary libraries
  ```
  from ultralytics import YOLO
  import cv2
  from matplotlib import pyplot as plt
  from PIL import Image
  from google.colab import output
  from base64 import b64decode
  ```
  
- Used YOLO pre-trained model for inference
  ```
  model = YOLO('yolov8n.pt')
  ```

- Opened webcam using `cv2.VideoCapture()`  
- Applied YOLO detections on each frame in real time  
- Displayed detections instantly in a live window  

- Use this function to take the snapshot
  ```
  def take_photo(filename = 'snapshot.jpg'):
    js = """
   async function takePhoto() {
     const div = document.createElement('div');
     const capture = document.createElement('button');
     capture.textContent = 'Capture';
     div.appendChild(capture);
     document.body.appendChild(div);


     const video = document.createElement('video');
     video.style.display = 'block';
     const stream = await navigator.mediaDevices.getUserMedia({video: true});
     document.body.appendChild(video);
     video.srcObject = stream;
     await video.play();


     // Resize window
     google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);


     // Wait for Capture button
     await new Promise((resolve) => capture.onclick = resolve);


     const canvas = document.createElement('canvas');
     canvas.width = video.videoWidth;
     canvas.height = video.videoHeight;
     canvas.getContext('2d').drawImage(video, 0, 0);
     stream.getTracks().forEach(track => track.stop());
     const imgData = canvas.toDataURL('image/jpeg').split(',')[1];
     div.remove();
     return imgData;
   }
   takePhoto();
   """
  ```



**Key Libraries:**  
- OpenCV  
- YOLO  

---

## 🏋️ Training YOLO on a Custom Dataset  

**Goal:** Understand the workflow of training YOLO.  

**Steps Taken:**  
- ```
  !pip install ultralytics --quiet
  from ultralytics import YOLO
  ```
  
- Used a small dataset (subset of COCO / custom images)  
- Prepared dataset with annotations and labels  
- Configured YOLO training for a few epochs  
- Evaluated the model on validation data  

**Concepts Learned:**  
- Dataset preparation  
- Training pipeline  
- Evaluation metrics (accuracy, loss curves)  

**Key Libraries:**  
- YOLO Training Scripts  
- Python  

---

## 🌐 Deployment with Streamlit + Ngrok_Token   

**Goal:** Deploy the YOLO model in an interactive web app.  

**Steps Taken:**  
- Built a **Streamlit** with a simple UI  
- Allowed users to upload images or videos  
- Displayed YOLO detection results inside the app  
- Used **Ngrok** to generate a public link with authentication token  

**Key Libraries:**  
- Streamlit  
- Ngrok  

---

## 📦 Installation  

Install the required packages before running the notebooks or app:  

```bash
!pip install ultralytics
!pip install opencv-python
!pip install pillow
!pip install streamlit
!pip install pyngrok
