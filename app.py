import streamlit as st
import os
from dotenv import load_dotenv
import base64
from PIL import Image
import io
import tempfile
import cv2
from groq import Groq

# Load environment variables
load_dotenv()

def setup_page():
    st.set_page_config(
        page_title="Media Analysis Bot",
        page_icon="🎥",
        layout="wide"
    )
    st.title("Media Analysis Bot")
    st.write("Upload an image or video and ask questions about it!")

def load_api_key():
    api_key = st.secrets["GROQ_API_KEY"]
    if not api_key:
        api_key = st.text_input("Enter your Groq API key:", type="password")
        if not api_key:
            st.warning("Please enter your Groq API key to continue.")
            st.stop()
    return api_key

def encode_image(file):
    try:
        # Reset file pointer to beginning
        file.seek(0)
        # Read file content
        file_content = file.read()
        # Encode to base64
        base64_encoded = base64.b64encode(file_content).decode('utf-8')
        return base64_encoded
    except Exception as e:
        st.error(f"Error encoding file: {str(e)}")
        return None

def process_video_frame(video_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(video_file.read())
            video_path = tmp_file.name

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Get middle frame
        middle_frame = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
        ret, frame = cap.read()
        
        if ret:
            # Convert frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            # Convert to base64
            base64_encoded = base64.b64encode(buffer).decode('utf-8')
            
            # Convert BGR to RGB for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            display_image = Image.fromarray(frame_rgb)
            
            cap.release()
            os.unlink(video_path)
            return base64_encoded, display_image
        else:
            cap.release()
            os.unlink(video_path)
            return None, None
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        return None, None

def analyze_with_groq(client, base64_image, query):
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Please analyze this image and answer: {query}"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
        
        with st.spinner("Analyzing with Groq..."):
            response = client.chat.completions.create(
                messages=messages,
                model="llama-3.2-11b-vision-preview",
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        return None

def main():
    setup_page()
    
    # Get API key
    api_key = load_api_key()
    
    # Initialize Groq client
    client = Groq(api_key=api_key)
    
    # File upload section
    st.write("### Upload Media")
    media_type = st.radio("Select media type:", ["Image", "Video"])
    
    file_types = ["jpg", "jpeg", "png"] if media_type == "Image" else ["mp4"]
    uploaded_file = st.file_uploader(
        f"Choose {media_type.lower()} file",
        type=file_types,
        help="Maximum recommended file size: 5MB"
    )
    
    # Query input
    query = st.text_input(
        "What would you like to know about the media?",
        help="Ask a specific question about the content"
    )
    
    if uploaded_file and query:
        # Check file size
        file_size = uploaded_file.size / (1024 * 1024)  # Convert to MB
        if file_size > 10:
            st.error("File is too large! Please upload a file smaller than 10MB.")
            return
        
        try:
            # Process media based on type
            if media_type == "Image":
                st.image(uploaded_file, caption="Uploaded Image")
                base64_image = encode_image(uploaded_file)
            else:
                st.write("Processing video frame...")
                base64_image, display_frame = process_video_frame(uploaded_file)
                if display_frame:
                    st.image(display_frame, caption="Analyzed Video Frame (Middle Frame)")
            
            if base64_image:
                # Analyze with Groq
                analysis = analyze_with_groq(client, base64_image, query)
                
                if analysis:
                    st.success("Analysis Complete!")
                    st.write("### Analysis Results")
                    st.write(analysis)
            else:
                st.error("Failed to process media file.")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    
    # Tips section
    with st.expander("Tips for better results"):
        st.write("""
        - Use images smaller than 5MB for best results
        - For images: Upload clear, well-lit images in JPG/PNG format
        - For videos: The middle frame will be analyzed, so ensure key content is visible
        - Ask specific, focused questions about the content
        - If you get an error, try reducing the file size or simplifying your query
        """)
    
    # Footer
    st.write("---")
    st.write("Thank you for using the Media Analysis Bot!")

if __name__ == "__main__":
    main()
