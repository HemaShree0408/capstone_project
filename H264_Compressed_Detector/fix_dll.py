import zipfile
import os
import shutil

# 1. The zip file you placed
zip_path = r"C:\Windows\System32\zlibwapi_x86-64.zip"

# Check if the zip exists
if not os.path.exists(zip_path):
    print(f"❌ File not found at: {zip_path}")
    print("Please make sure you put the ZIP there!")
else:
    # 2. Extract only the DLL
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # We look for the one in the x64 folder
        try:
            # Note: This checks the internal path in the zip
            dll_internal_name = "zlib123dllx64/dll_x64/zlibwapi.dll" 
            # If the zip structure is different, it might just be:
            for name in zip_ref.namelist():
                if name.endswith("zlibwapi.dll") and "x64" in name:
                    dll_internal_name = name
                    break
            
            print(f"📦 Extracting: {dll_internal_name}")
            zip_ref.extract(dll_internal_name, ".")
            
            # 3. Move it to the root project folder
            source_path = os.path.join(".", dll_internal_name)
            destination_path = "zlibwapi.dll"
            
            shutil.move(source_path, destination_path)
            print("✅ Successfully moved zlibwapi.dll to your project!")
            print("Done! Now run 'python train.py' and your RTX 4090 will work!")
            
        except Exception as e:
            print(f"❌ Extraction error: {e}")
            print("Try extracting it manually if this script fails.")
