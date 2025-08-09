import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from skimage.transform import resize
import os
from datetime import datetime

# PAGE CONFIGURATION
st.markdown("""
<style>
    /* App background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #f5f7fa 100%);
        color: #000000;
        font-family: 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    }

    /* Title Styling */
    .stTitle h1 {
        font-weight: 700;
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 0.5rem;
        background: linear-gradient(to right, #3498db, #2c3e50);
        background-clip: text;
        -webkit-background-clip: text;
        color: transparent;
        -webkit-text-fill-color: transparent;
    }

    /* Custom box styling */
    .custom-box {
        background: white !important;
        color: black !important;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.25rem 0;
        box-shadow: 0 6px 18px rgba(0,0,0,0.08);
        border-left: 5px solid #3498db;
    }
    
    .custom-title {
        color: #3498db;
        font-weight: 700;
        font-size: 1.25rem;
        margin-bottom: 0.75rem;
    }

    /* Selectbox label & File uploader label */
    .stSelectbox label,
    .stFileUploader label {
        color: #3498db !important;
        font-weight: bold !important;
        font-size: 20px !important;
    }

    /* Inputs & buttons */
    .stTextInput > div > div > input,
    .stSelectbox > div > div,
    .stFileUploader > div > div {
        background-color: white;
        color: #000000;
        border: 1px solid #B6B09F;
        border-radius: 8px;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(to right, #3498db, #2980b9);
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        font-weight: bold;
    }
    .stButton > button:hover {
        background: linear-gradient(to right, #2980b9, #3498db);
    }

    /* Download Button */
    .stDownloadButton button {
        background: #3498db !important;
        color: #f5f7fa !important;
        font-weight: bold !important;
    }
    .stDownloadButton button:hover {
        background: #CC0000 !important;
        color: white !important;
    }

    /* Force specific divs text to black */
    div.st-ak.st-al.st-bd.st-be.st-bf.st-as.st-bg.st-bh.st-ar.st-bi.st-bj.st-bk.st-bl {
        color: black !important;
        box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    }
    div.st-an.st-ao.st-ap.st-aq.st-ak.st-ar.st-am.st-as.st-at.st-au.st-av.st-aw.st-ax.st-ay.st-az.st-b0.st-b1.st-b2.st-b3.st-b4.st-b5.st-b6.st-b7.st-b8.st-b9.st-ba.st-bb.st-bc {
        background-color: white !important;
    }
    svg[data-baseweb="icon"] {
    fill: #2980b9; 
    }
            
    section.st-emotion-cache-1erivf3.e16xj5sw0 {
        background-color: white !important;
        color: black !important;
        border-radius: 8px !important;
        border: 1px solid #ccc !important;
        padding: 1rem !important;
    }

    section.st-emotion-cache-1erivf3.e16xj5sw0 * {
        color: black !important;
    }

    div.st-emotion-cache-j7qwjs {
        background-color: white !important;
        color: black !important;
        border-radius: 6px !important;
        padding: 0.5rem !important;
    }

    div.st-emotion-cache-j7qwjs * {
        color: black !important;
    }

    small.st-emotion-cache-c8ta4l.ejh2rmr0 {
        background-color: white !important;
        color: black !important;
        padding: 0.25rem 0.5rem !important;
        border-radius: 4px !important;
    }
    button.st-emotion-cache-ktz07o.eacrzsi2 {
        background-color: #3498db !important;
        color: #f5f7fa !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.5rem 1rem !important;
        cursor: pointer !important;
    }

    button.st-emotion-cache-ktz07o.eacrzsi2:hover {
        background-color: #3498db !important;
        
    }      
 
</style>

<script>
document.addEventListener("DOMContentLoaded", function() {
    function changeLabelColor() {
        document.querySelectorAll("*").forEach(function(el) {
            if (el.textContent.trim().toLowerCase() === "select task" ||
                el.textContent.trim().toLowerCase() === "upload mri image") {
                el.style.color = "#3498db";
                el.style.fontWeight = "bold";
                el.style.fontSize = "20px";
            }
        });
    }
    changeLabelColor();
    const observer = new MutationObserver(changeLabelColor);
    observer.observe(document.body, { childList: true, subtree: true });
});
</script>
""", unsafe_allow_html=True)

# CUSTOM COMPONENTS
def show_box(message, title_type="Info", icon="ℹ️"):
    icons = {
        "Info": "ℹ️",
        "Success": "✅",
        "Error": "❌",
        "Welcome": "📌",
        "Result": "🔍",
        "Confidence": "📊",
        "Saved": "💾",
        "Upload Required": "📤"
    }
    icon = icons.get(title_type, icon)
    title = f"{icon} {title_type}"
    st.markdown(
        f"""
        <div class='custom-box'>
            <div class='custom-title'>
                {title}
            </div>
            <div style='line-height: 1.6;'>
                {message}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# MODEL FUNCTIONS
def preprocess_segmentation_image(image):
    if isinstance(image, Image.Image):
        image = np.array(image)
    image = resize(image, (256, 256), mode='constant', preserve_range=True)
    if len(image.shape) == 2:
        image = np.stack((image,) * 3, axis=-1)
    elif image.shape[-1] == 4:
        image = image[..., :3]
    return image / 255.0

def preprocess_classification_image(image):
    if isinstance(image, Image.Image):
        image = np.array(image)
    image = resize(image, (224, 224), mode='constant', preserve_range=True)
    if len(image.shape) == 2:
        image = np.stack((image,) * 3, axis=-1)
    elif image.shape[-1] == 4:
        image = image[..., :3]
    return image / 255.0

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - ((2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth))

def combined_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def iou_metric(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + 1e-6) / (union + 1e-6)

@st.cache_resource
def load_segmentation_model():
    try:
        model = tf.keras.models.load_model(
            r"C:\Users\Ayakhaled\Downloads\Classification & Segmentatin Models\BrainTumor_Segmentation_Unet.h5", # upload your segmentation model here
            custom_objects={'combined_loss': combined_loss, 'iou_metric': iou_metric}
        )
        show_box("Segmentation model loaded successfully.", "Success")
        return model
    except Exception as e:
        show_box(f"Failed to load segmentation model: {e}", "Error")
        return None

@st.cache_resource
def load_classification_model():
    try:
        model = tf.keras.models.load_model(
            r"C:\Users\Ayakhaled\Downloads\Classification & Segmentatin Models\BrainTumor_classification_model.h5" # upload your classification model here
        )
        show_box("Classification model loaded successfully.", "Success")
        return model
    except Exception as e:
        show_box(f"Failed to load classification model: {e}", "Error")
        return None

# MAIN APP INTERFACE
st.markdown("""
<div class='stTitle'>
    <h1>🧠 Advanced Brain Tumor Analysis</h1>
</div>
""", unsafe_allow_html=True)

show_box("Welcome to our advanced MRI analysis tool. Upload an image and select a task to proceed.", "Welcome")

# Sidebar
with st.sidebar:
    st.title("🔬 Project Information")
    st.markdown("---")
    st.markdown("""
    **Model Architecture**  
    🧠 Classification: CNN  
    🎯 Segmentation: U-Net  
    """)
    st.markdown("---")
    st.markdown(f"**Last Run**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.markdown("**Developer**: Aya Khaled Farouk")
    st.markdown("---")
    st.markdown("""
    <div class='sidebar-footer'>
        Medical Imaging AI • Research Project
    </div>
    """, unsafe_allow_html=True)

task = st.selectbox("Select Task", ["Segmentation", "Classification"], key="task_select")

uploaded_file = st.file_uploader("Upload MRI Image", type=["tif", "png", "jpg", "jpeg"], key="file_uploader")

output_dir = "predictions"
os.makedirs(output_dir, exist_ok=True)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded MRI Image", use_container_width=True)
    except Exception as e:
        show_box(f"Error loading image: {e}", "Error")
        st.stop()

    if task == "Segmentation":
        processed_image = preprocess_segmentation_image(image)
        processed_image_batch = np.expand_dims(processed_image, axis=0)

        segmentation_model = load_segmentation_model()
        if segmentation_model is None:
            st.stop()

        try:
            with st.spinner("Performing segmentation..."):
                pred_mask = segmentation_model.predict(processed_image_batch, verbose=0)[0]
                pred_mask_binary = (pred_mask > 0.5).astype(np.float32)

            st.subheader("Segmentation Result")
            st.image(pred_mask_binary[..., 0], caption="Predicted Segmentation Mask", use_container_width=True, clamp=True)

            # تحويل الماسك إلى صورة للتنزيل
            mask_image = Image.fromarray((pred_mask_binary[..., 0] * 255).astype(np.uint8))
            
            # حفظ الصورة وتنزيلها عند النقر على زر واحد
            output_image_path = os.path.join(output_dir, f"mask_{uploaded_file.name}")
            if st.download_button(
                "Download Mask",
                data=mask_image.tobytes(),
                file_name=os.path.basename(output_image_path),
                mime="image/png",
                help="Click to download and save the mask"
            ):
                mask_image.save(output_image_path)
                show_box(f"Mask has been saved to {output_image_path}", "Saved")

        except Exception as e:
            show_box(f"Error during segmentation prediction: {e}", "Error")

    elif task == "Classification":
        processed_image = preprocess_classification_image(image)
        processed_image_batch = np.expand_dims(processed_image, axis=0)

        classification_model = load_classification_model()
        if classification_model is None:
            st.stop()

        try:
            with st.spinner("Performing classification..."):
                pred_prob = classification_model.predict(processed_image_batch, verbose=0)[0][0]
                pred_class = "Tumor" if pred_prob > 0.5 else "No Tumor"
                confidence = pred_prob if pred_prob > 0.5 else 1 - pred_prob

            st.subheader("Classification Result")
            show_box(f"Prediction: {pred_class}", "Result")
            show_box(f"Confidence: {confidence:.2%}", "Confidence")

            # إنشاء محتوى النتيجة للتنزيل
            result_content = f"Prediction: {pred_class}\nConfidence: {confidence:.2%}"
            output_text_path = os.path.join(output_dir, f"result_{uploaded_file.name}.txt")
            
            if st.download_button(
                "Download Result",
                data=result_content,
                file_name=os.path.basename(output_text_path),
                mime="text/plain",
                help="Click to download and save the result"
            ):
                with open(output_text_path, "w", encoding="utf-8") as f:
                    f.write(result_content)
                show_box(f"Result has been saved to {output_text_path}", "Saved")

        except Exception as e:
            show_box(f"Error during classification prediction: {e}", "Error")
else:
    show_box("Please upload an MRI image to get started.", "Upload Required")

