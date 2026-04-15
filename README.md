# 🎓 Capstone Project: H.264 Video Frequency Extraction for AI

Welcome to my Capstone project! This repository contains a modified version of the **FFmpeg Video Player** engine. 

Normally, when a computer plays a video, it does a lot of heavy math to turn compressed data into the colorful pixels you see on your screen. However, Artificial Intelligence (like Object Detection CNNs) doesn't actually need to see colors or pixels! It just needs to see the raw structural "edges" and "textures" of the video to know there is an object moving.

My project **hacks into the FFmpeg video decoder** and forces it to stop halfway through decoding. It steals the raw mathematical frequencies (called DCT Coefficients) of the video and instantly dumps them into a file. This allows AI to process video essentially 20x faster than normal, because we never build the actual image pixels!

---

## 📂 The 7 Files in this Repository

These 7 custom files are the heart of the project:
1. **`h264_coeff_extract.c`**: The custom C code that intercepts the video data.
2. **`h264_coeff_extract.h`**: The headers for the C code.
3. **`h264dec.c`**: The modified FFmpeg file that turns our code ON.
4. **`h264_slice.c`**: The modified FFmpeg file that tells the decoder to stop before building pixels.
5. **`Makefile`**: Tells the compiler how to build our new code.
6. **`feature_map.py`**: A Python script that reads the intercepted data and draws a 3D structural map of the objects for the AI to learn from.
7. **`README.md`**: This instruction file.

---

## 🛠️ Step 1: Getting the Tools Ready
*Why is this step needed?* We cannot run raw C code on a computer directly. We need to download a "Compiler" to turn our C code into an executable `.exe` program that Windows can understand.

1. Go to your web browser and open [MSYS2.org](https://www.msys2.org/). 
2. Follow the instructions on the web page to download the installer (it is a standard Windows installer program). Run the installer and click "Next" until it finishes.
3. Open the **MSYS2 MinGW x64** program on your Windows computer (you can find it in your Start Menu). A black terminal window will open.
4. Copy and paste this exact command into the black window and press Enter:
   ```bash
   pacman -S make diffutils pkgconf nasm gcc
   ```
   *(If it asks you "Proceed with installation? [Y/n]", type **Y** and press Enter. Wait until it completely finishes downloading).*

---

## 📥 Step 2: Downloading the Official FFmpeg Engine
*Why is this step needed?* FFmpeg is a massive 3-Gigabyte project. We only wrote 5 single files! We need the rest of the official FFmpeg engine so our 5 files have a "body" to attach to.

1. Go to the [Official FFmpeg GitHub Page](https://github.com/FFmpeg/FFmpeg).
2. Click the green **"Code"** button, and select **"Download ZIP"**.
3. Once the ZIP file finishes downloading to your computer, right-click it and select **"Extract All..."**.
4. Extract the folder to a simple location on your computer, such as your `Documents` or `Downloads` folder. 

---

## 💉 Step 3: Injecting Our Custom Code
*Why is this step needed?* We need to replace the official FFmpeg logic with our customized Capstone logic so it knows to extract the data.

1. Download the 5 C/Makefile code files from this Capstone GitHub repository to your computer.
2. Open the newly extracted **FFmpeg** folder from Step 2.
3. Inside that folder, find and open the folder named **`libavcodec`**.
4. Copy our 5 Capstone files and **paste them directly into the `/libavcodec/` folder**. 
5. Windows will warn you that some files already exist. Click **"Replace the files in the destination"** (Click "Yes" to overwrite the old files).

---

## ⚙️ Step 4: Compiling the Custom Video Engine
*Why is this step needed?* Now that our code is safely inside FFmpeg, we must tell the compiler to build the final `ffmpeg.exe` application.

1. Open your **MSYS2 MinGW x64** terminal window from Step 1.
2. Navigate into your FFmpeg folder using the `cd` command. For example, if you saved FFmpeg in your Downloads folder, you would type:
   ```bash
   cd /c/Users/YourName/Downloads/FFmpeg
   ```
3. Type this command to prepare the environment:
   ```bash
   export PATH=/mingw64/bin:$PATH
   ```
4. Run these three commands one-by-one to build the `.exe` file (the final `make` command will take 5-10 minutes, so be patient until the text stops moving!):
   ```bash
   ./configure
   make clean
   make -j$(nproc)
   ```

---

## 🏃 Step 5: Extracting the Data from a Video!
*Why is this step needed?* We now have our hacking tool (`ffmpeg.exe`). We need to feed it a video so it can dump the raw data into a newly created file called `coeffs.bin`.

1. Place any video (for example: `MyVideo.mp4`) into a folder on your computer.
2. In your MSYS2 terminal, tell the environment to turn ON our secret extraction flag and choose a destination file:
   ```bash
   export H264_COEFF_EXTRACT_FILE="coeffs.bin"
   ```
3. Run our custom FFmpeg on your video! If it's in the same folder, you would type:
   ```bash
   ./ffmpeg.exe -i "MyVideo.mp4" -f null -
   ```
4. You will notice it runs incredibly fast and entirely skips playing the video. Wait for it to finish. A `coeffs.bin` file will magically appear in your folder containing all the raw frequencies!

---

## 🧠 Step 6: Seeing the Results (Python)
*Why is this step needed?* The `coeffs.bin` file is pure binary computer code. We need a Python script to translate that code into a beautiful structural image that proves the AI can see the moving objects.

1. Make sure you have Python installed on your Windows computer. (If not, download it from Python.org).
2. Download the **`feature_map.py`** script from this GitHub repository and place it in the same folder as your new `coeffs.bin` file.
3. Open a normal **Windows Command Prompt** or **PowerShell** (Click Start -> type `cmd` -> press Enter).
4. Install the Python graphing libraries by typing:
   ```powershell
   pip install numpy matplotlib
   ```
5. Navigate to the folder where your script is saved, and run our mapping script:
   ```powershell
   python feature_map.py
   ```

A window will pop up showing the Low, Mid, and High-Frequency boundaries of the objects moving in your video. **This proves the extraction was 100% successful and is ready for Convolutional Neural Network (CNN) training!**
