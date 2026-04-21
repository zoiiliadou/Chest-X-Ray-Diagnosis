from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import uvicorn
import tensorflow as tf
import numpy as np
from PIL import Image, ImageStat
import io
import sqlite3
import pydicom
from datetime import datetime
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

import base64
import matplotlib.cm as cm

app = FastAPI(title="Pneumonia Detection API")
templates = Jinja2Templates(directory="Templates")

def make_gradcam_base64(img_array, original_img, model, last_conv_layer_name="out_relu"):
    # Generate heatmap
    grad_model = tf.keras.models.Model(model.inputs, [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        outputs = grad_model(img_array)
        last_conv_layer_output, preds = outputs[0], outputs[1]
        
        if isinstance(preds, list):
            preds = preds[0]
        if isinstance(last_conv_layer_output, list):
            last_conv_layer_output = last_conv_layer_output[0]
            
        class_channel = preds[:, 0]
        
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    heatmap = last_conv_layer_output[0] @ tf.expand_dims(pooled_grads, axis=-1)
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    
    # Map to colors using matplotlib
    jet = cm.get_cmap('jet')
    heatmap_color = jet(heatmap)[:, :, :3]
    
    # Resize using PIL
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap_color))
    heatmap_img = heatmap_img.resize((original_img.shape[1], original_img.shape[0]), Image.Resampling.LANCZOS)
    heatmap_color_resized = np.array(heatmap_img) / 255.0
    
    # Superimpose
    superimposed_img = heatmap_color_resized * 0.4 + original_img * 0.6
    superimposed_img = np.clip(superimposed_img, 0, 1)
    
    # Convert to base64
    result_img = Image.fromarray(np.uint8(255 * superimposed_img))
    buffered = io.BytesIO()
    result_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return "data:image/png;base64," + img_str

# Load the medical model
MODEL = tf.keras.models.load_model('pneumonia_model.h5')

# Load generic MobileNetV2 (OOD Bouncer Model)
try:
    print("Loading OOD Bouncer Model (MobileNetV2 ImageNet)...")
    BOUNCER_MODEL = MobileNetV2(weights='imagenet')
    print("OOD Bouncer Model loaded successfully.")
except Exception as e:
    print(f"Failed to load OOD Model: {e}")
    BOUNCER_MODEL = None

def init_db():
    try:
        conn = sqlite3.connect('xray_history.db')
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                timestamp TEXT,
                prediction TEXT,
                confidence REAL,
                has_heatmap BOOLEAN
            )
        ''')
        conn.commit()
    except Exception as e:
        print(f"Database initialization error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

init_db()

def check_if_xray_heuristic(image):
    """
    Checks if an image resembles an X-ray by analyzing its colors and contrast.
    The 'image' must be a PIL.Image object in RGB format.
    """
    img_array = np.array(image).astype(np.float32)
    
    # 1. Check if it's truly Grayscale
    if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
        channels = img_array[:, :, :3]
        max_c = np.max(channels, axis=2)
        min_c = np.min(channels, axis=2)
        diffs = max_c - min_c
        
        # Stricter threshold (0.5%) to catch even a thin colored line (e.g., in graphs).
        color_pixels_ratio = np.mean(diffs > 20)
        if color_pixels_ratio > 0.005:
            return False, "Η εικόνα δεν αναγνωρίζεται ως έγκυρη ακτινογραφία θώρακος. Παρακαλούμε βεβαιωθείτε ότι το αρχείο αφορά την κατάλληλη ιατρική απεικόνιση."

    # 2. Brightness Check (White Background)
    # Scientific plots and documents often have >40% pure white background.
    # X-rays only have white areas around bones (much smaller percentage).
    gray_mean = np.mean(img_array[:, :, :3], axis=2) if len(img_array.shape) == 3 else img_array
    white_ratio = np.mean(gray_mean > 225)
    if white_ratio > 0.35: # If more than 35% of the image is pure white
        return False, "Η εικόνα δεν αναγνωρίζεται ως έγκυρη ακτινογραφία θώρακος. Παρακαλούμε βεβαιωθείτε ότι το αρχείο αφορά την κατάλληλη ιατρική απεικόνιση."

    # 3. Contrast Check
    # X-rays have high variance due to black and white (bone) regions.
    # Low deviation indicates a very "flat" image (e.g., plain white/gray background, document).
    std_dev = np.std(img_array)
    if std_dev < 25: 
         return False, "Η εικόνα δεν αναγνωρίζεται ως έγκυρη ακτινογραφία θώρακος. Παρακαλούμε βεβαιωθείτε ότι το αρχείο αφορά την κατάλληλη ιατρική απεικόνιση."

    return True, "Φαίνεται ΟΚ"

def check_if_xray_ai(image):
    """
    Uses pre-trained MobileNetV2 (ImageNet) to detect if the image
    is something "known" but irrelevant (like grayscale screenshots, documents, PDFs).
    """
    if BOUNCER_MODEL is None:
        return True, "ΟΚ"
        
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized.convert('RGB'))
    img_array = np.expand_dims(img_array, axis=0)
    
    # Preprocess specifically for MobileNetV2
    x = preprocess_input(img_array.astype(np.float32))
    
    preds = BOUNCER_MODEL.predict(x)
    decoded = decode_predictions(preds, top=3)[0]
    
    # List of possible ImageNet classes indicating a document, computer, or graph
    blacklist_keywords = ['web_site', 'envelope', 'comic_book', 'book_jacket', 'menu', 
                          'monitor', 'screen', 'television', 'laptop', 'mouse', 'keyboard', 
                          'desktop_computer', 'cellular_telephone', 'puzzle', 'crossword_puzzle',
                          'notebook', 'paper', 'binder', 'oscilloscope', 'maze', 'rule', 'measuring_cup']
                          
    for i in range(3):
        class_id, class_name, confidence = decoded[i]
        class_name_lower = class_name.lower()
        if confidence > 0.15: # If relatively confident it saw something from the blacklist
            for keyword in blacklist_keywords:
                if keyword in class_name_lower:
                    return False, "Η εικόνα δεν αναγνωρίζεται ως έγκυρη ακτινογραφία θώρακος. Παρακαλούμε βεβαιωθείτε ότι το αρχείο αφορά την κατάλληλη ιατρική απεικόνιση."
                    
    # If ANY known natural/anatomical/everyday object is detected with high confidence (>60%), reject
    top_class_id, top_class_name, top_conf = decoded[0]
    if top_conf > 0.60:
         return False, "Η εικόνα δεν αναγνωρίζεται ως έγκυρη ακτινογραφία θώρακος. Παρακαλούμε βεβαιωθείτε ότι το αρχείο αφορά την κατάλληλη ιατρική απεικόνιση."

    return True, "ΟΚ"

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # 1st Check: Basic characteristics (color and contrast)
    is_xray, msg = check_if_xray_heuristic(img)
    if not is_xray:
        return False, msg, None

    # 2nd Check: AI (MobileNetV2 ImageNet) to detect B&W documents/screenshots
    is_xray, msg = check_if_xray_ai(img)
    if not is_xray:
        return False, msg, None
    
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return True, "OK", img_array

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    
    # Get the result of the preprocess check
    is_xray, error_msg, processed_image = preprocess_image(image_bytes)
    
    if not is_xray:
        return {
            "prediction": "Άσχετη Εικόνα",
            "confidence": 0.0,
            "error_msg": error_msg, # Dynamic error message
            "raw_score": 0.0,
            "threshold_used": 0.0
        }
    
    predictions = MODEL.predict(processed_image)
    score = float(predictions[0][0])
    
    threshold = 0.92 
    result = "Pneumonia" if score > threshold else "Normal"
    confidence = score if result == "Pneumonia" else (1 - score)

    # 3rd Check: Explainable AI - Calculate Grad-CAM Heatmap
    heatmap_b64 = ""
    try:
        heatmap_b64 = make_gradcam_base64(processed_image, processed_image[0], MODEL, "out_relu")
    except Exception as e:
        print(f"Grad-CAM Failed: {e}")

    has_heatmap = (heatmap_b64 != "")
    
    # Save to Database
    try:
        conn = sqlite3.connect('xray_history.db')
        cursor = conn.cursor()
        timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        safe_filename = file.filename if file.filename else "unknown_file"
        cursor.execute('''
            INSERT INTO analyses (filename, timestamp, prediction, confidence, has_heatmap)
            VALUES (?, ?, ?, ?, ?)
        ''', (safe_filename, timestamp, result, round(confidence, 4), has_heatmap))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Σφάλμα εγγραφής στη βάση: {e}")

    return {
        "prediction": result,
        "confidence": round(confidence, 4),
        "raw_score": round(score, 4),
        "threshold_used": threshold,
        "heatmap": heatmap_b64
    }

@app.get("/history")
async def get_history(pin: str = ""):
    if pin != "1234":
        return {"status": "unauthorized", "message": "Απαιτείται κωδικός ιατρού για την προβολή του ιστορικού."}
        
    try:
        conn = sqlite3.connect('xray_history.db')
        cursor = conn.cursor()
        cursor.execute("SELECT id, filename, timestamp, prediction, confidence, has_heatmap FROM analyses ORDER BY id DESC LIMIT 50")
        rows = cursor.fetchall()
        conn.close()
        
        history = []
        for row in rows:
            history.append({
                "id": row[0],
                "filename": row[1],
                "timestamp": row[2],
                "prediction": row[3],
                "confidence": row[4],
                "has_heatmap": bool(row[5])
            })
        return {"status": "success", "data": history}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/delete-history")
async def delete_history(pin: str = ""):
    if pin != "1234":
        return {"status": "unauthorized", "message": "Απαιτείται κωδικός ιατρού για τη διαγραφή."}
    try:
        conn = sqlite3.connect('xray_history.db')
        cursor = conn.cursor()
        cursor.execute("DELETE FROM analyses")
        conn.commit()
        conn.close()
        return {"status": "success", "message": "Το ιστορικό διαγράφηκε επιτυχώς."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/predict-dicom")
async def predict_dicom(file: UploadFile = File(...)):
    contents = await file.read()
    
    # 1. Read the DICOM file
    try:
        dicom_bytes = io.BytesIO(contents)
        ds = pydicom.dcmread(dicom_bytes)
    except Exception as e:
        return {"prediction": "Άσχετη Εικόνα", "confidence": 0.0, "error_msg": f"Μη έγκυρο αρχείο DICOM: {str(e)}", "raw_score": 0.0, "threshold_used": 0.0}

    # 2. Extract Metadata
    def get_tag(ds, tag, default="N/A"):
        try:
            val = getattr(ds, tag, None)
            return str(val) if val is not None else default
        except:
            return default

    metadata = {
        "PatientName":    get_tag(ds, "PatientName"),
        "PatientAge":     get_tag(ds, "PatientAge"),
        "PatientSex":     get_tag(ds, "PatientSex"),
        "StudyDate":      get_tag(ds, "StudyDate"),
        "Modality":       get_tag(ds, "Modality"),
        "InstitutionName": get_tag(ds, "InstitutionName"),
        "StudyDescription": get_tag(ds, "StudyDescription"),
    }

    # 3. Convert pixel data to PIL image
    try:
        pixel_array = ds.pixel_array.astype(np.float32)
        pixel_array = np.nan_to_num(pixel_array, nan=0.0, posinf=255.0, neginf=0.0)
        # Normalize to 0-255
        pmin, pmax = pixel_array.min(), pixel_array.max()
        if pmax > pmin:
            pixel_array = (pixel_array - pmin) / (pmax - pmin) * 255.0
        pixel_array = np.clip(pixel_array, 0, 255).astype(np.uint8)
        
        if len(pixel_array.shape) == 2:
            image = Image.fromarray(pixel_array).convert("RGB")
        elif len(pixel_array.shape) == 3:
            image = Image.fromarray(pixel_array[:, :, 0]).convert("RGB")
        else:
            raise ValueError("Unsupported pixel array shape")
    except Exception as e:
        return {"prediction": "Άσχετη Εικόνα", "confidence": 0.0, "error_msg": f"Σφάλμα επεξεργασίας DICOM pixels: {str(e)}", "raw_score": 0.0, "threshold_used": 0.0}

    # 4. Run the same AI pipeline as standard predict
    if not check_if_xray_heuristic(image):
        return {"prediction": "Άσχετη Εικόνα", "confidence": 0.0, "error_msg": "Το αρχείο DICOM δεν φαίνεται να αντιστοιχεί σε ακτινογραφία θώρακος.", "raw_score": 0.0, "threshold_used": 0.0, "metadata": metadata}

    img_resized = image.resize((224, 224))
    img_array  = np.array(img_resized, dtype=np.float32) / 255.0
    processed  = np.expand_dims(img_array, axis=0)

    predictions = MODEL.predict(processed)
    score = float(predictions[0][0])
    threshold = 0.92
    result = "Pneumonia" if score > threshold else "Normal"
    confidence = score if result == "Pneumonia" else (1 - score)

    heatmap_b64 = ""
    try:
        heatmap_b64 = make_gradcam_base64(processed, processed[0], MODEL, "out_relu")
    except Exception as e:
        print(f"DICOM Grad-CAM Failed: {e}")

    # 5. Convert original image to Base64 for UI display
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    original_b64 = "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode()

    # 6. Save to database
    has_heatmap = (heatmap_b64 != "")
    try:
        conn = sqlite3.connect('xray_history.db')
        cursor = conn.cursor()
        timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        cursor.execute('''
            INSERT INTO analyses (filename, timestamp, prediction, confidence, has_heatmap)
            VALUES (?, ?, ?, ?, ?)
        ''', (file.filename or "dicom_file.dcm", timestamp, result, round(confidence, 4), has_heatmap))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DICOM DB write error: {e}")

    return {
        "prediction": result,
        "confidence": round(confidence, 4),
        "raw_score": round(score, 4),
        "threshold_used": threshold,
        "heatmap": heatmap_b64,
        "original_image": original_b64,
        "metadata": metadata
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)