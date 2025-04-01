# **Deepfake Image Generation with DCGAN**

This project implements a **Deep Convolutional Generative Adversarial Network (DCGAN)** using **PyTorch** to generate deepfake images. The primary goal is to train a **generator network** that creates realistic-looking synthetic images and a **discriminator network** that distinguishes between real and generated images. Both networks improve through adversarial training, leading to more convincing deepfake images.

> **Disclaimer:** This project is strictly for **research and educational purposes**. It is not intended for malicious use, deceptive content creation, or unethical applications. Please use responsibly.

---

## **1️⃣ Features**
- **DCGAN Architecture**: Uses deep convolutional layers for generating high-quality images.
- **PyTorch Implementation**: Utilizes `torchvision` for image handling and model building.
- **GPU Support**: Optimized for CUDA-enabled GPUs.
- **Adversarial Training**: Generator and discriminator compete to improve image quality.
- **Checkpointing**: Saves trained models (`G.pkl` and `D.pkl`) for reuse.

---

## **2️⃣ Installation & Requirements**
### **📌 Prerequisites**
- Python **3.7+**
- PyTorch **1.10+**
- Torchvision **0.11+**
- CUDA **(Optional: For GPU acceleration)**
- OpenCV
- Other common dependencies like `os`

### **💻 Installation**
Run the following command to install dependencies:
```bash
pip install torch torchvision opencv-python
```

---

## **3️⃣ Dataset Preparation**
Your dataset should be structured as follows:

```
dataset/
    class1/
        image1.jpg
        image2.png
        ...
    class2/
        image1.jpg
        image2.jpeg
        ...
    ...
```
- Each subdirectory within `dataset/` represents a class or category of images.
- **Important:** The quality and diversity of the dataset significantly impact GAN performance.
- Ensure `dataset_path` in the script is correctly set to your dataset location.

---

## **4️⃣ Model Architecture**
### **🌀 Generator**
- Takes a **100-dimensional noise vector** and outputs a **64x64x3 image**.
- Uses `ConvTranspose2d`, **BatchNorm**, and `ReLU` activation layers.
- The final layer applies a **Tanh** activation function.

### **🔎 Discriminator**
- Takes a **64x64x3 image** as input and outputs a probability of being real or fake.
- Uses `Conv2d`, **LeakyReLU**, and `BatchNorm` layers.
- The final layer applies a **Sigmoid** activation function.

![cd899bd9-9c48-46ad-91e5-28c86ac86830](https://github.com/user-attachments/assets/2d707c2a-216c-4258-a3dc-b9ec0fd41bf5)


---

## **5️⃣ Training Details**
| Parameter        | Value      |
|-----------------|-----------|
| **Optimizer**   | RMSprop    |
| **Learning Rate** | 0.0005  |
| **Batch Size**  | 256       |
| **Loss Function** | Binary Cross Entropy (BCELoss) |
| **Epochs**      | 200       |
| **Device**      | GPU (if available) |

---

## **6️⃣ Usage**
### **🔹 1. Prepare Dataset**
Ensure your dataset is properly formatted and update `dataset_path` in the script.

### **🔹 2. Train the Model**
Run the training script:
```bash
python your_script_name.py
```
Replace `your_script_name.py` with the actual filename.

- During training, sample images are saved in the `output/` directory.
- Trained model weights are saved as:
  - `G.pkl` (Generator)
  - `D.pkl` (Discriminator)

### **🔹 3. Track Training Progress**
- Sample images are generated after each epoch and saved in `output/`.
- Training progress logs display **Generator (G) and Discriminator (D) loss values**.

---

## **7️⃣ Results**
- As training progresses, generated images improve in realism.
- The saved sample images in `output/` help track improvement.
- Example results:
  - **Epoch 1:** Noisy, unrealistic images
  - **Epoch 50:** Some recognizable features
  - **Epoch 200:** High-quality deepfake images

---

## **8️⃣ Future Improvements**
🔹 **Enhance Image Quality**: Experiment with architectures, loss functions, and training methods.  
🔹 **Conditional GANs (cGANs)**: Generate images with controlled attributes like identity, pose, or expression.  
🔹 **Face Swapping**: Integrate generated images into a face-swapping pipeline.  
🔹 **Evaluation Metrics**: Implement **Inception Score (IS)** or **Fréchet Inception Distance (FID)** for quality assessment.  
🔹 **Data Augmentation**: Apply random transformations to real images during training.

---

## **9️⃣ Screenshots**

<img width="1440" alt="Screenshot 2025-03-20 at 11 34 09 PM" src="https://github.com/user-attachments/assets/560793bf-e108-4cc8-a36f-c4a4b279a6a7" />

<img width="1440" alt="Screenshot 2025-03-20 at 11 35 43 PM" src="https://github.com/user-attachments/assets/733f0b34-22b8-460e-bd15-998d47393ec5" />

---

## **🔗 References**
- Radford, A., Metz, L., & Chintala, S. (2015). **Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks**. *arXiv preprint arXiv:1511.06434*.  
- Goodfellow, I., et al. (2014). **Generative Adversarial Networks**. *NeurIPS Conference*.

---

## 🚀 **Conclusion**
This project demonstrates how **DCGANs** can generate realistic deepfake images. With future improvements, the model can be enhanced for higher-quality, more controllable outputs.

---
