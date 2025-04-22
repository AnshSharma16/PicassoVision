import streamlit as st
import tensorflow as tf
import numpy as np
import PIL.Image
import time
import pickle
import cv2
import os
from matplotlib import pyplot as plt

# Set these to reduce TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set page configuration with a custom theme
st.set_page_config(
    page_title="Neural Pencil Sketch Generator", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a more artistic look
import streamlit as st

# Apply custom CSS for a professional dark theme with color accents
st.markdown("""
<style>
    /* Dark theme base with colorful accents */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #e2e8f0;
    }
    
    /* Main Header - Colorful yet professional */
    .main-header {
        font-family: 'Montserrat', sans-serif;
        font-size: 3.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        letter-spacing: -0.02em;
        text-align: center;
    }
    
    /* Sub Header with glass effect */
    .sub-header {
        font-family: 'Montserrat', sans-serif;
        font-size: 1.5rem;
        color: #94a3b8;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
        letter-spacing: 0.05em;
        padding: 0.5rem;
        border-bottom: 1px solid rgba(148, 163, 184, 0.2);
    }
    
    /* Professional buttons with glow effect */
    .stButton > button {
        background: linear-gradient(90deg, #0ea5e9, #6366f1);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.75rem 1.75rem;
        font-weight: 600;
        letter-spacing: 0.025em;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
        text-transform: uppercase;
        font-size: 0.9rem;
    }
    
    .stButton > button:hover {
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5);
        transform: translateY(-2px);
    }
    
    /* Override Streamlit's default text styling */
    .stMarkdown, .stText, p, div {
        color: #e2e8f0;
    }
    
    /* Custom card-like containers */
    .css-1r6slb0, .css-1y4p8pa {  /* Streamlit containers */
        background: rgba(30, 41, 59, 0.7);
        border-radius: 10px;
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border: 1px solid rgba(148, 163, 184, 0.1);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        margin: 1rem 0;
    }
    
    /* Style for inputs and selectboxes */
    div[data-baseweb="input"], div[data-baseweb="select"] {
        background-color: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 6px;
    }
    
    div[data-baseweb="input"]:focus-within, div[data-baseweb="select"]:focus-within {
        border-color: #818cf8;
        box-shadow: 0 0 0 2px rgba(129, 140, 248, 0.2);
    }
    
    /* Input text color */
    .stTextInput input, .stSelectbox select {
        color: #e2e8f0 !important;
    }
    
    /* Slider styling */
    .stSlider div[role="slider"] {
        background-color: #818cf8;
    }
    
    .stSlider div[role="slider"] > div {
        background-image: linear-gradient(90deg, #0ea5e9, #6366f1);
    }
    
    /* Decorative neon accents */
    .decorator {
        position: fixed;
        z-index: -1;
        border-radius: 50%;
        filter: blur(60px);
        opacity: 0.15;
    }
    
    .decorator-1 {
        top: 10%;
        left: 10%;
        width: 250px;
        height: 250px;
        background: #38bdf8;
    }
    
    .decorator-2 {
        bottom: 10%;
        right: 10%;
        width: 300px;
        height: 300px;
        background: #c084fc;
    }
    
    /* Header decoration */
    .header-glow {
        width: 100%;
        height: 1px;
        background: linear-gradient(90deg, transparent, #818cf8, transparent);
        margin: 0 auto 3rem auto;
    }
    
    /* For data tables */
    .stDataFrame {
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 8px;
        overflow: hidden;
    }
    
    .stDataFrame table {
        background-color: #1e293b;
    }
    
    .stDataFrame th {
        background-color: #0f172a;
        color: #e2e8f0;
        padding: 0.75rem 1rem;
        border-bottom: 1px solid rgba(148, 163, 184, 0.2);
    }
    
    .stDataFrame td {
        color: #e2e8f0;
        padding: 0.75rem 1rem;
        border-bottom: 1px solid rgba(148, 163, 184, 0.1);
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1e293b;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #475569;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #64748b;
    }
</style>

<!-- Add decorative elements -->
<div class="decorator decorator-1"></div>
<div class="decorator decorator-2"></div>
""", unsafe_allow_html=True)

# Example of using the custom classes
st.markdown('<h1 class="main-header">ANALYTICS DASHBOARD</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="sub-header">Professional Insights & Data Visualization</h3>', unsafe_allow_html=True)
st.markdown('<div class="header-glow"></div>', unsafe_allow_html=True)

# Demo content
st.subheader("Interactive Dashboard")
st.write("This is your professionally styled Streamlit app with a dark theme and vibrant color accents.")

col1, col2 = st.columns(2)
with col1:
    st.button("ANALYZE DATA")
    
with col2:
    st.text_input("Search metrics")

# Add example chart or data
import numpy as np
import pandas as pd

# Sample data
chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['Alpha', 'Beta', 'Gamma'])

st.line_chart(chart_data)

# Example metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Revenue", "$12,345", "+4.5%")
col2.metric("Users", "1,234", "+12.3%")
col3.metric("Conversion", "5.7%", "-0.5%")
col4.metric("Avg. Time", "2m 34s", "+1.2%")

# Load the model config
try:
    with open('model.pkl', 'rb') as f:
        model_config = pickle.load(f)
    model_loaded = True
except:
    model_loaded = False
    st.warning("Model configuration file not found. Using default parameters.")
    model_config = {
        'content_layers': ['block4_conv2'],
        'style_layers': ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1'],
        'style_weight': 300,
        'content_weight': 0.5,
        'epochs': 10,
        'steps_per_epoch': 50,
        'optimizer_params': {
            'learning_rate': 0.02,
            'beta_1': 0.99,
            'epsilon': 1e-1
        }
    }


# Define functions
def tensor_to_image(tensor):
    """Converts a tensor to a PIL Image."""
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def load_img(img_data):
    """Loads an image from uploaded data and converts it to a tensor."""
    max_dim = 512
    img = tf.image.decode_image(img_data, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def create_pencil_sketch_style(content_image):
    # Convert tensor to numpy array for OpenCV processing
    img = content_image[0].numpy()
    img = (img * 255).astype(np.uint8)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur to reduce noise
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Create edge detection using Laplacian
    edges = cv2.Laplacian(gray_blur, cv2.CV_8U, ksize=5)

    # Invert the image for pencil sketch effect
    inverted = 255 - edges

    # Apply threshold to make edges more prominent
    _, sketch = cv2.threshold(inverted, 180, 255, cv2.THRESH_BINARY)

    # Convert back to RGB (still grayscale in appearance)
    sketch_rgb = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)

    # Convert back to tensor format
    sketch_tensor = tf.convert_to_tensor(sketch_rgb, dtype=tf.float32) / 255.0
    sketch_tensor = sketch_tensor[tf.newaxis, :]

    return sketch_tensor


def vgg_layers(layer_names):
    """Creates a VGG model that returns a list of intermediate output values."""
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model

def gram_matrix(input_tensor):
    """Calculate the Gram matrix of an input tensor."""
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)

def clip_0_1(image):
    """Clips image tensor values to between 0 and 1."""
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def style_content_loss(outputs, style_targets, content_targets, style_weight, content_weight):
    """Calculate the total style and content loss."""
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2)
                            for name in style_outputs.keys()])
    style_loss *= style_weight / len(style_outputs)

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2)
                               for name in content_outputs.keys()])
    content_loss *= content_weight / len(content_outputs)
    total_loss = style_loss + content_loss
    return total_loss

class StyleContentModel(tf.keras.models.Model):
    """Model that extracts style and content features."""
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        """Expects normalized inputs in range [0,1]"""
        inputs = inputs*255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                        outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output)
                       for style_output in style_outputs]

        content_dict = {content_name: value
                      for content_name, value
                      in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value
                    for style_name, value
                    in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}

@tf.function()
def train_step(image, style_targets, content_targets, optimizer, style_weight, content_weight, extractor):
    """Update the target image."""
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs, style_targets, content_targets, style_weight, content_weight)
        loss += tf.image.total_variation(image) * 30

    grad = tape.gradient(loss, image)
    optimizer.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))


def to_dark_pencil_sketch(image_tensor):
    """Convert the final output to a darker, more realistic pencil sketch."""
    # Convert tensor to numpy array
    img = image_tensor[0].numpy()
    img = (img * 255).astype(np.uint8)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply adaptive thresholding to create stronger line effect
    adaptive_threshold = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # Invert to get the black lines on white paper effect
    inverted_sketch = 255 - adaptive_threshold

    # Enhance contrast
    _, enhanced_sketch = cv2.threshold(inverted_sketch, 150, 255, cv2.THRESH_BINARY)

    # Blur slightly to soften the sketch and make it more natural
    sketch_blurred = cv2.GaussianBlur(enhanced_sketch, (3, 3), 0)

    # Convert back to RGB (still grayscale in appearance)
    sketch_rgb = cv2.cvtColor(sketch_blurred, cv2.COLOR_GRAY2RGB)

    # Convert back to tensor format
    final_sketch = tf.convert_to_tensor(sketch_rgb, dtype=tf.float32) / 255.0
    final_sketch = final_sketch[tf.newaxis, :]

    return final_sketch

def to_pencil_dodge_burn(image_tensor):
    """Create a realistic pencil sketch using dodge and burn technique."""
    # Convert tensor to numpy array
    img = image_tensor[0].numpy()
    img = (img * 255).astype(np.uint8)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Invert the grayscale image
    inverted = 255 - gray

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(inverted, (21, 21), 0)

    # Invert blurred image
    inverted_blur = 255 - blur

    # Divide gray by inverted blur image (dodge effect)
    pencil_sketch = cv2.divide(gray, inverted_blur, scale=256.0)

    # Enhance the sketch by adjusting contrast
    alpha = 1.5  # Contrast control (1.0-3.0)
    beta = 10    # Brightness control (0-100)
    enhanced_sketch = cv2.convertScaleAbs(pencil_sketch, alpha=alpha, beta=beta)

    # Make the sketch darker
    _, darker_sketch = cv2.threshold(enhanced_sketch, 160, 255, cv2.THRESH_BINARY)

    # Convert back to RGB (still grayscale in appearance)
    sketch_rgb = cv2.cvtColor(darker_sketch, cv2.COLOR_GRAY2RGB)

    # Convert back to tensor format
    final_sketch = tf.convert_to_tensor(sketch_rgb, dtype=tf.float32) / 255.0
    final_sketch = final_sketch[tf.newaxis, :]

    return final_sketch

def run_sketch_conversion(content_image, progress_bar, status_text, epochs, steps_per_epoch, style_weight, content_weight):
    """Run sketch conversion on the given content image with progress updates."""
    # Create pencil sketch style reference
    style_image = create_pencil_sketch_style(content_image)
    
    # Initialize model
    content_layers = model_config['content_layers']
    style_layers = model_config['style_layers']
    
    extractor = StyleContentModel(style_layers, content_layers)

    # Set targets
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']

    # Initialize image to optimize
    image = tf.Variable(content_image)

    # Set optimization parameters
    opt = tf.optimizers.Adam(
        learning_rate=model_config['optimizer_params']['learning_rate'], 
        beta_1=model_config['optimizer_params']['beta_1'], 
        epsilon=model_config['optimizer_params']['epsilon']
    )

    # Run optimization with progress updates
    total_steps = epochs * steps_per_epoch
    step_counter = 0
    
    for n in range(epochs):
        for m in range(steps_per_epoch):
            train_step(image, style_targets, content_targets, opt, style_weight, content_weight, extractor)
            step_counter += 1
            progress = step_counter / total_steps
            progress_bar.progress(progress)
            status_text.text(f"Processing: {int(progress*100)}% (Epoch {n+1}/{epochs}, Step {m+1}/{steps_per_epoch})")
    
    status_text.text("Generating final sketches...")
    
    # Convert to final sketches
    adaptive_sketch = to_dark_pencil_sketch(image.read_value())
    dodge_burn_sketch = to_pencil_dodge_burn(image.read_value())
    
    # Return all versions
    return (tensor_to_image(image.read_value()[0]),
            tensor_to_image(adaptive_sketch[0]),
            tensor_to_image(dodge_burn_sketch[0]))

# Sidebar with artistic header
st.sidebar.markdown("<h2 class='sub-header'>✏ Sketch Settings</h2>", unsafe_allow_html=True)

# Parameter sliders
epochs = st.sidebar.slider("Detail Level (Epochs)", min_value=1, max_value=20, value=model_config['epochs'], 
                         help="More epochs = better quality but slower processing")
steps_per_epoch = st.sidebar.slider("Refinement Steps", min_value=10, max_value=100, 
                                  value=model_config['steps_per_epoch'],
                                  help="More steps = better quality but slower processing")
style_weight = st.sidebar.slider("Sketch Intensity", min_value=100, max_value=600, 
                              value=model_config['style_weight'],
                              help="Higher value = stronger sketch effect")
content_weight = st.sidebar.slider("Content Preservation", min_value=0.1, max_value=2.0, 
                                value=model_config['content_weight'], step=0.1,
                                help="Higher value = more faithful to original content")

# File uploader with a custom container
st.markdown("""
<div style="padding: 1.5rem; border-radius: 0.7rem; background: linear-gradient(145deg, #f8f9fa, #e9ecef); box-shadow: 5px 5px 10px #d1d9e6, -5px -5px 10px #ffffff;">
    <h3 style="text-align: center; color: #2c3e50; margin-bottom: 1rem;">Upload Your Image</h3>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image to transform", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Create columns with artistic styling
    col1, col2 = st.columns(2)
    
    # Display uploaded image
    img = PIL.Image.open(uploaded_file)
    col1.markdown("<h3 class='sub-header'>Original Image</h3>", unsafe_allow_html=True)
    col1.image(img, use_column_width=True)
    
    # Create placeholder for preview
    col2.markdown("<h3 class='sub-header'>Preview</h3>", unsafe_allow_html=True)
    preview_placeholder = col2.empty()
    
    # Button to start processing with centered layout
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        start_button = st.button("✨ Transform to Pencil Sketch ✨", use_container_width=True)
    
    if start_button:
        # Progress indicators
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<h3 class='sub-header' style='text-align: center;'>Processing Your Artwork</h3>", unsafe_allow_html=True)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process image
        try:
            status_text.text("Preparing your image...")
            img_data = uploaded_file.getvalue()
            content_image = load_img(img_data)
            
            # Create pencil sketch reference for preview
            sketch_reference = create_pencil_sketch_style(content_image)
            preview_placeholder.image(tensor_to_image(sketch_reference[0]), use_column_width=True, caption="Sketch Reference")
            
            # Run the conversion
            status_text.text("Starting neural style transfer...")
            styled_image, adaptive_sketch, dodge_burn_sketch = run_sketch_conversion(
                content_image, progress_bar, status_text, 
                epochs, steps_per_epoch, style_weight, content_weight
            )
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.text("Processing complete! ✅")
            
            # Display results with tabs
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("<h2 class='sub-header' style='text-align: center;'>Your Sketches</h2>", unsafe_allow_html=True)
            
            tabs = st.tabs(["Neural Style Transfer", "Classic Sketch", "Dodge & Burn Sketch"])
            
            with tabs[0]:
                st.image(styled_image, use_container_width=True)
                buf = io.BytesIO()
                styled_image.save(buf, format="PNG")
                byte_im = buf.getvalue()
                st.download_button(
                    label="Download Neural Style Transfer",
                    data=byte_im,
                    file_name="neural_style_transfer.png",
                    mime="image/png"
                )
                
            with tabs[1]:
                st.image(adaptive_sketch, use_container_width=True)
                buf = io.BytesIO()
                adaptive_sketch.save(buf, format="PNG")
                byte_im = buf.getvalue()
                st.download_button(
                    label="Download Classic Sketch",
                    data=byte_im,
                    file_name="classic_sketch.png",
                    mime="image/png"
                )
                
            with tabs[2]:
                st.image(dodge_burn_sketch, use_container_width=True)
                buf = io.BytesIO()
                dodge_burn_sketch.save(buf, format="PNG")
                byte_im = buf.getvalue()
                st.download_button(
                    label="Download Dodge & Burn Sketch",
                    data=byte_im,
                    file_name="dodge_burn_sketch.png",
                    mime="image/png"
                )
                
        except Exception as e:
            st.error(f"Error processing image: {e}")
            progress_bar.empty()
            status_text.text("Processing failed.")
    
    # Add a separator before the about section
    st.markdown("<hr>", unsafe_allow_html=True)

# Display information about the app
st.sidebar.markdown("---")
st.sidebar.markdown("<h3 class='sub-header'>About This App</h3>", unsafe_allow_html=True)
st.sidebar.markdown("""
<div style="background: linear-gradient(145deg, #f8f9fa, #e9ecef); padding: 1rem; border-radius: 0.7rem; box-shadow: inset 2px 2px 5px #d1d9e6, inset -2px -2px 5px #ffffff;">
    <p style="font-size: 0.9rem; color: #2c3e50;">
        This app uses neural style transfer powered by VGG19 to transform your photos into artistic pencil sketches.
        <br><br>
        The process combines deep learning techniques with traditional image processing to create multiple sketch styles.
        <br><br>
        <b>How it works:</b>
        <ul>
            <li>Neural Style Transfer extracts content from your image</li>
            <li>It applies the style of a pencil sketch</li>
            <li>Post-processing enhances the sketch-like qualities</li>
        </ul>
    </p>
</div>
""", unsafe_allow_html=True)

# Import missing module
import io

# Only show if model not loaded
if not model_loaded:
    st.sidebar.warning("Running with default parameters. For best results, upload the model.pkl file.")