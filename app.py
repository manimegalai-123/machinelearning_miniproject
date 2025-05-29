import streamlit as st
import numpy as np
import zipfile
import tempfile
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import plotly.express as px
import pandas as pd
import shutil

# Constants
IMG_SIZE = 224
CLASSES = ['0', 'R', 'MR', 'MRMS', 'MS', 'S']
CLASS_DESCRIPTIONS = {
    '0': 'Healthy (No Yellow Rust)',
    'R': 'Resistant (Very Low Infection)',
    'MR': 'Moderately Resistant (Low Infection)',
    'MRMS': 'Moderately Resistant to Moderately Susceptible',
    'MS': 'Moderately Susceptible (High Infection)',
    'S': 'Susceptible (Very High Infection)'
}

# Set page config
st.set_page_config(
    page_title="Yellow Rust Classifier",
    page_icon="🌾",
    layout="wide"
)

@st.cache_resource
def load_trained_model():
    """
    Load the trained model from NEWMODEL.zip file with multiple fallback methods
    """
    model_path = "NEWMODEL.zip"
    
    # Check if zip file exists
    if not os.path.exists(model_path):
        st.error("❌ NEWMODEL.zip file not found! Please ensure the model file is in the same directory.")
        return None
    
    st.info("🔄 Loading model...")
    
    # Method 1: Try direct loading (for .h5 models saved as zip)
    try:
        st.write("Attempting Method 1: Direct loading from zip...")
        model = load_model(model_path)
        st.success("✅ Model loaded successfully using direct method!")
        return model
    except Exception as e1:
        st.write(f"Method 1 failed: {str(e1)}")
    
    # Method 2: Extract and load
    try:
        st.write("Attempting Method 2: Extract and load...")
        
        with zipfile.ZipFile(model_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            st.write(f"Files in zip: {file_list[:10]}...")  # Show first 10 files
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract all files
                zip_ref.extractall(temp_dir)
                
                # Look for different model formats
                extracted_items = os.listdir(temp_dir)
                st.write(f"Extracted items: {extracted_items}")
                
                # Check for SavedModel format (has saved_model.pb)
                for item in extracted_items:
                    item_path = os.path.join(temp_dir, item)
                    
                    if os.path.isdir(item_path):
                        # Check if this directory contains saved_model.pb
                        if 'saved_model.pb' in os.listdir(item_path):
                            st.write(f"Found SavedModel in: {item}")
                            model = tf.keras.models.load_model(item_path)
                            st.success("✅ SavedModel loaded successfully!")
                            return model
                    
                    # Check for .h5 files
                    elif item.endswith('.h5'):
                        st.write(f"Found H5 model: {item}")
                        model = load_model(item_path)
                        st.success("✅ H5 model loaded successfully!")
                        return model
                
                # If no direct model found, check subdirectories
                for root, dirs, files in os.walk(temp_dir):
                    if 'saved_model.pb' in files:
                        st.write(f"Found SavedModel in subdirectory: {root}")
                        model = tf.keras.models.load_model(root)
                        st.success("✅ SavedModel from subdirectory loaded successfully!")
                        return model
                    
                    for file in files:
                        if file.endswith('.h5'):
                            h5_path = os.path.join(root, file)
                            st.write(f"Found H5 model in subdirectory: {h5_path}")
                            model = load_model(h5_path)
                            st.success("✅ H5 model from subdirectory loaded successfully!")
                            return model
                
    except Exception as e2:
        st.write(f"Method 2 failed: {str(e2)}")
    
    # Method 3: Try loading without compilation
    try:
        st.write("Attempting Method 3: Loading without compilation...")
        model = load_model(model_path, compile=False)
        st.success("✅ Model loaded without compilation!")
        return model
    except Exception as e3:
        st.write(f"Method 3 failed: {str(e3)}")
    
    # Method 4: Try TFSMLayer (for TensorFlow SavedModel in zip)
    try:
        st.write("Attempting Method 4: TFSMLayer approach...")
        
        # Extract to a permanent location for TFSMLayer
        extract_dir = "temp_model_extract"
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)
        
        with zipfile.ZipFile(model_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Find the actual model directory
        model_dir = None
        for root, dirs, files in os.walk(extract_dir):
            if 'saved_model.pb' in files:
                model_dir = root
                break
        
        if model_dir:
            # Create a wrapper model using TFSMLayer
            tfsm_layer = keras.layers.TFSMLayer(model_dir, call_endpoint='serving_default')
            
            # Create input based on expected shape
            inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
            outputs = tfsm_layer(inputs)
            
            # Handle different output formats
            if isinstance(outputs, dict):
                # If output is a dictionary, take the first value
                outputs = list(outputs.values())[0]
            
            model = keras.Model(inputs=inputs, outputs=outputs)
            st.success("✅ TFSMLayer model loaded successfully!")
            return model
        
    except Exception as e4:
        st.write(f"Method 4 failed: {str(e4)}")
        # Clean up
        if os.path.exists("temp_model_extract"):
            shutil.rmtree("temp_model_extract")
    
    st.error("❌ All loading methods failed. Please check your model file format.")
    return None

def preprocess_image(image):
    """
    Preprocess the uploaded image for prediction
    """
    try:
        # Convert to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size
        img = image.resize((IMG_SIZE, IMG_SIZE))
        
        # Convert to array and normalize
        img_array = img_to_array(img) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    except Exception as e:
        st.error(f"❌ Error preprocessing image: {str(e)}")
        return None

def make_prediction(model, img_array):
    """
    Make prediction on the preprocessed image with improved error handling
    """
    try:
        # Make prediction
        prediction = model.predict(img_array, verbose=0)
        
        # Handle different output formats
        if isinstance(prediction, dict):
            # If prediction is a dictionary, get the output tensor
            prediction = list(prediction.values())[0]
        
        if hasattr(prediction, 'numpy'):
            prediction = prediction.numpy()
        
        # Ensure prediction is in the right format
        if len(prediction.shape) > 2:
            prediction = prediction.reshape(prediction.shape[0], -1)
        
        # Apply softmax if values don't sum to ~1 (indicating raw logits)
        if not np.isclose(np.sum(prediction[0]), 1.0, atol=0.1):
            prediction = tf.nn.softmax(prediction).numpy()
        
        # Ensure we have the right number of classes
        if prediction.shape[1] != len(CLASSES):
            st.warning(f"⚠️ Model output shape {prediction.shape} doesn't match expected classes {len(CLASSES)}")
            # Pad or truncate as needed
            if prediction.shape[1] < len(CLASSES):
                padding = np.zeros((1, len(CLASSES) - prediction.shape[1]))
                prediction = np.concatenate([prediction, padding], axis=1)
            else:
                prediction = prediction[:, :len(CLASSES)]
        
        return prediction
        
    except Exception as e:
        st.error(f"❌ Error making prediction: {str(e)}")
        st.error(f"Model type: {type(model)}")
        st.error(f"Input shape: {img_array.shape}")
        
        # Try to get more info about the model
        try:
            st.write(f"Model input shape: {model.input_shape}")
            st.write(f"Model output shape: {model.output_shape}")
        except:
            pass
        
        return None

def display_prediction_results(prediction):
    """
    Display prediction results with confidence scores and charts
    """
    class_idx = np.argmax(prediction)
    confidence = np.max(prediction)
    predicted_class = CLASSES[class_idx]
    
    # Main prediction result
    st.success(f"🎯 **Prediction:** {predicted_class} - {CLASS_DESCRIPTIONS[predicted_class]}")
    st.info(f"📊 **Confidence:** {confidence:.2%}")
    
    # Create two columns for detailed results
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📈 Confidence Scores")
        
        # Create dataframe for all predictions
        results_df = pd.DataFrame({
            'Class': CLASSES,
            'Description': [CLASS_DESCRIPTIONS[cls] for cls in CLASSES],
            'Confidence': prediction[0],
            'Percentage': prediction[0] * 100
        }).sort_values('Confidence', ascending=False)
        
        # Display as table
        st.dataframe(
            results_df[['Class', 'Description', 'Percentage']].round(2),
            use_container_width=True,
            hide_index=True
        )
    
    with col2:
        st.subheader("📊 Prediction Chart")
        
        # Create bar chart
        fig = px.bar(
            results_df,
            x='Class',
            y='Percentage',
            color='Percentage',
            color_continuous_scale='RdYlGn_r',
            title="Confidence Scores by Class",
            labels={'Percentage': 'Confidence (%)'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk assessment
    st.subheader("🔍 Risk Assessment")
    
    if predicted_class == '0':
        st.success("✅ **Low Risk:** The wheat leaf appears healthy with no signs of yellow rust.")
        st.write("**Recommendation:** Continue regular monitoring and maintain good agricultural practices.")
    elif predicted_class in ['R', 'MR']:
        st.warning("⚠️ **Low to Moderate Risk:** Early signs of yellow rust detected, but plant shows resistance.")
        st.write("**Recommendation:** Monitor closely and consider preventive fungicide application if conditions favor disease development.")
    elif predicted_class == 'MRMS':
        st.warning("⚠️ **Moderate Risk:** Moderate yellow rust infection detected.")
        st.write("**Recommendation:** Apply appropriate fungicide treatment and monitor spread to adjacent plants.")
    elif predicted_class in ['MS', 'S']:
        st.error("🚨 **High Risk:** Significant yellow rust infection detected.")
        st.write("**Recommendation:** Immediate fungicide treatment required. Consider resistant varieties for future planting.")

def main():
    # Header
    st.title("🌾 Yellow Rust Classifier")
    st.markdown("### AI-Powered Wheat Disease Detection")
    st.write("Upload a wheat leaf image to predict yellow rust infection stage and severity.")
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About Yellow Rust")
        st.write("""
        **Yellow Rust** (Puccinia striiformis) is a serious fungal disease affecting wheat crops.
        
        **Classification Categories:**
        - **0**: Healthy leaf
        - **R**: Resistant (minimal infection)
        - **MR**: Moderately Resistant
        - **MRMS**: Moderately Resistant to Moderately Susceptible
        - **MS**: Moderately Susceptible
        - **S**: Susceptible (severe infection)
        """)
        
        st.header("📋 Usage Instructions")
        st.write("""
        1. Upload a clear image of wheat leaf
        2. Ensure good lighting and focus
        3. Image should show leaf surface clearly
        4. Wait for AI analysis
        5. Review results and recommendations
        """)
    
    # Load model
    with st.spinner("🔄 Loading AI model..."):
        model = load_trained_model()
    
    if model is None:
        st.error("Failed to load the model. Please check if NEWMODEL.zip exists and is valid.")
        st.stop()
    
    st.success("✅ Model loaded successfully!")
    
    # Display model info if available
    try:
        st.info(f"Model input shape: {model.input_shape}")
        st.info(f"Model output shape: {model.output_shape}")
    except:
        pass
    
    # File uploader
    st.subheader("📤 Upload Wheat Leaf Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        help="Supported formats: JPG, JPEG, PNG, BMP, TIFF"
    )
    
    if uploaded_file is not None:
        try:
            # Load and display image
            image = Image.open(uploaded_file)
            
            # Display original image
            st.subheader("📷 Uploaded Image")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)
            
            # Image info
            st.write(f"**Image Details:** {image.size[0]}x{image.size[1]} pixels, Mode: {image.mode}")
            
            # Preprocess image
            with st.spinner("🔄 Preprocessing image..."):
                img_array = preprocess_image(image)
            
            if img_array is not None:
                # Make prediction
                with st.spinner("🤖 Analyzing image with AI..."):
                    prediction = make_prediction(model, img_array)
                
                if prediction is not None:
                    st.markdown("---")
                    st.subheader("🎯 Analysis Results")
                    
                    # Display results
                    display_prediction_results(prediction)
                    
                    # Additional options
                    st.markdown("---")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🔄 Analyze Another Image"):
                            st.rerun()
                    
                    with col2:
                        # Option to download results
                        results_text = f"""
Yellow Rust Analysis Report
==========================
Image: {uploaded_file.name}
Prediction: {CLASSES[np.argmax(prediction)]} - {CLASS_DESCRIPTIONS[CLASSES[np.argmax(prediction)]]}
Confidence: {np.max(prediction):.2%}

Detailed Scores:
{chr(10).join([f'{cls}: {pred:.2%}' for cls, pred in zip(CLASSES, prediction[0])])}
                        """
                        st.download_button(
                            "📄 Download Report",
                            results_text,
                            file_name=f"yellow_rust_report_{uploaded_file.name}.txt",
                            mime="text/plain"
                        )
        
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
    
    else:
        # Show example or placeholder
        st.info("👆 Please upload a wheat leaf image to begin analysis.")
        
        # Tips for best results
        st.subheader("💡 Tips for Best Results")
        st.write("""
        - Use clear, well-lit images
        - Focus on the leaf surface
        - Avoid blurry or dark images
        - Include the affected area in the frame
        - Higher resolution images work better
        """)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        <p>🌾 Yellow Rust Classifier | Powered by Deep Learning & Streamlit</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
