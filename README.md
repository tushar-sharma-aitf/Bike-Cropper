# 🏍️ Motorcycle Image Cropping System
**Advanced AI-powered motorcycle image cropping tool with three specialized cropping modes**
---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage Guide](#-usage-guide)


---

## 🌟 Overview

The **Motorcycle Image Cropping System** is an intelligent, AI-powered tool designed to automatically detect and crop motorcycle images with pixel-perfect precision. Built with YOLOv8 for object detection and Gradio for an intuitive web interface, this system offers three specialized cropping modes optimized for different use cases.

### Why This Tool?

- 🎯 **Automatic Detection**: No manual selection needed - AI detects motorcycles instantly
- 🔧 **Multiple Crop Modes**: Three specialized modes for different requirements
- 🎨 **Professional Results**: Ultra-tight cropping with minimal padding
- 🌐 **Bilingual Support**: Full English and Japanese language support
- ⚡ **Fast Processing**: Real-time cropping with instant results
- 📦 **Easy to Use**: Web-based interface accessible from any browser

---

## ✨ Features

### 🤖 Core Features

| Feature | Description |
|---------|-------------|
| **YOLOv8 Detection** | State-of-the-art object detection for accurate motorcycle identification |
| **Three Crop Modes** | Specialized modes: Extreme Tight, Custom, and Solid Color Background |
| **Zero Horizontal Padding** | Perfect front-to-rear cropping with 0px side margins |
| **Bounding Box Visualization** | Visual feedback showing detection accuracy |
| **Custom Padding Control** | Adjustable padding with 90% reduction for ultra-tight crops |
| **Resize Options** | Optional output resizing to target dimensions |
| **Ground Removal** | Automatic floor/ground removal with background fill |
| **Bilingual Interface** | Full English (en) and Japanese (ja) language support |

### 🎨 Cropping Modes

#### Type 1: Extreme Tight Crop
- **Horizontal Padding**: 0px (zero)
- **Vertical Padding**: 3px
- **Best For**: Product catalogs, listings, tight compositions
- **Speed**: Instant

#### Type 2: Extreme Custom Crop
- **Horizontal Padding**: Adjustable (10% of slider value)
- **Vertical Padding**: Adjustable (20% of slider value)
- **Resize**: Optional target width setting
- **Best For**: Custom requirements, specific dimensions
- **Speed**: Instant

#### Type 3: Solid Color Background
- **Horizontal Padding**: 0px
- **Vertical Padding**: 3px
- **Ground Removal**: Yes (crops to tyre level)
- **Background Fill**: Solid color matching original background
- **Best For**: E-commerce, professional listings, clean backgrounds
- **Speed**: Fast



## 📦 Installation

### Prerequisites

- **Python**: 3.8 or higher
- **pip**: Latest version
- **GPU**: Optional (CUDA-enabled GPU for faster processing)

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 4 GB | 8 GB+ |
| **Storage** | 2 GB free | 5 GB+ free |
| **OS** | Windows 10/11, macOS 10.14+, Ubuntu 18.04+ | Latest version |

---

## 🚀 Quick Start

### Step 1: Clone the Repository
git clone https://github.com/yourusername/Bike-Cropper.git
cd Bike-Cropper


### Step 2: Install Dependencies
pip install -r requirements.txt


### Step 3: Run the Application
python app.py


### Step 4: Open in Browser
The application will automatically launch at:
http://127.0.0.1:7860/


---

## 💻 Usage Guide

### Basic Usage

1. **Launch Application**
python app.py

2. **Upload Image**
- Click "Upload Motorcycle Image"
- Select your motorcycle photo

3. **Choose Crop Mode**
- Navigate to desired tab (Type 1, 2, or 3)
- Adjust settings if using Type 2

4. **Process Image**
- Click the crop button
- View results instantly

5. **Download Results**
- Right-click on result image
- Select "Save image as..."

---



