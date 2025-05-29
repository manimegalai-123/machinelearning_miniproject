import streamlit as st
import numpy as np
import zipfile
import tempfile
import os
import tensorflow as tf
import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import plotly.express as px
import pandas as pd

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
    Load the trained model from NEWMODEL.zip file (TensorFlow SavedModel format)
    """
    try:
        # Check if zip file exists
        if not os.path.exists("NEWMODEL.zip"):
            st.error("❌ NEWMODEL.zip file not found! Please ensure the model file is in the same directory.")
            return None
        
        st.info("🔄 Loading TensorFlow SavedModel...")
        
        # Use TFSMLayer for TensorFlow SavedModel format as recommended in the error
        tfsm_layer = keras.layers.TFSMLayer("NEWMODEL.zip", call_endpoint='serving_default')
        
        # Create a functional model wrapper
        input_shape = (224, 224, 3)  # Based on your IMG_SIZE
        inputs = keras.Input(shape=input_shape)
        outputs = tfsm_layer(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        st.success("✅ TensorFlow SavedModel loaded successfully!")
        return model
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        
        # Try alternative approach - extract and load if it's actually a different format
        try:
            st.info("🔄 Trying alternative loading method...")
            
            # Extract the zip to see what's inside
            with zipfile.ZipFile("NEWMODEL.zip", 'r') as zip_ref:
                file_list = zip_ref.namelist()
                st.write(f"Files in zip: {file_list}")
                
                # Check for saved_model.pb (TensorFlow SavedModel)
                if any('saved_model.pb' in f for f in file_list):
                    # Extract to temporary directory
                    with tempfile.TemporaryDirectory() as temp_dir:
                        zip_ref.extractall(temp_dir)
                        
                        # Find the directory containing saved_model.pb
                        for root, dirs, files in os.walk(temp_dir):
                            if 'saved_model.pb' in files:
                                model = tf.keras.models.load_model(root)
                                st.success("✅ Extracted SavedModel loaded successfully!")
                                return model
                
                # Check for .h5 files
                h5_files = [f for f in file_list if f.endswith('.h5')]
                if h5_files:
                    h5_file = h5_files[0]
                    with tempfile.TemporaryDirectory() as temp_dir:
                        zip_ref.extract(h5_file, temp_dir)
                        h5_path = os.path.join(temp_dir, h5_file)
                        model = load_model(h5_path)
                        st.success("✅ H5 model loaded successfully!")
                        return model
                
                st.error("❌ Could not find compatible model format in the zip file")
                return None
                
        except Exception as alt_error:
            st.error(f"❌ Alternative loading also failed: {str(alt_error)}")
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
    Make prediction on the preprocessed image
    """
    try:
        # Make prediction
        prediction = model.predict(img_array, verbose=0)
        
        # Handle different output formats
        if hasattr(prediction, 'numpy'):
            prediction = prediction.numpy()
        
        # Ensure prediction is in the right format
        if len(prediction.shape) > 2:
            prediction = prediction.reshape(prediction.shape[0], -1)
        
        return prediction
        
    except Exception as e:
        st.error(f"❌ Error making prediction: {str(e)}")
        st.error(f"Model type: {type(model)}")
        st.error(f"Input shape: {img_array.shape}")
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
                            st.experimental_rerun()
                    
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
    
    else:
        # Show example or placeholder
        st.info("👆 Please upload a wheat leaf image to begin analysis.")
        
        # You can add example images here if you have them
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
