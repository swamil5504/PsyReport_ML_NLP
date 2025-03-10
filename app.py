import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import numpy as np
from streamlit_drawable_canvas import st_canvas
from streamlit_option_menu import option_menu
import os
from collections import Counter
from model_loaders import house_loader, tree_loader, person_loader
from model_loaders.house_loader import predict


# Sidebar with logo and navigation menu
with st.sidebar:
    st.image('images/logo.png', width=250)
    
    # Option menu for selecting prediction models
    selected = option_menu(
        'PsyReport',
        ['Overview', 'House Drawing Analysis', 'Tree Drawing Analysis', 'Person Drawing Analysis', 'NLP to be done'],
        icons=['house', 'activity', 'heart', 'person', 'clipboard2-heart'],
        default_index=0
    )



# Overview Page
if selected=="Overview":
    st.title("🧠 PsyReport: Unlocking Children's Minds Through Art")
    
    st.markdown(
        """
        ## 🌟 Why PsyReport?
        Children's drawings offer deep psychological insights, revealing their emotions, personality traits, and mental well-being.
        PsyReport is an **AI-powered psychological analysis tool** designed to interpret children's artwork through scientific methods.
        By analyzing drawings of **houses, trees, and people**, PsyReport provides valuable insights into a child's inner world.
        
        ## 🎯 What Does PsyReport Do?
        - **🔍 Emotional Analysis:** Detects hidden emotions such as joy, fear, anxiety, or stress in children's drawings.
        - **🧑‍⚕️ Mental Health Support:** Identifies early signs of emotional distress, aiding parents, teachers, and psychologists in providing timely intervention.
        - **🖌️ Personality Traits:** Assesses whether a child is introverted, extroverted, or experiencing emotional withdrawal.
        - **📜 AI-Powered Reports:** Generates insightful psychological assessments based on **deep learning and NLP models**.
        
        ## 🚀 How to Use PsyReport?
        1. **Select a Drawing Type:** Choose from House, Tree, or Person (HTP) analysis.
        2. **Upload or Create a Drawing:** Upload an image or use the interactive drawing canvas.
        3. **AI Interpretation:** Our advanced AI model analyzes the sketch for psychological patterns.
        4. **Receive a Detailed Report:** Get an AI-generated psychological analysis with professional insights.
        
        ### 🔬 Backed by Research
        - **HTP Test (House-Tree-Person):** A well-established projective test used in child psychology.
        - **Studies on Left-Behind Children (LBC):** Research highlights that LBC children often show emotional distress through artwork.
        - **AI in Mental Health:** Deep learning models significantly improve accuracy in psychological assessments of children's drawings.
        
        PsyReport bridges the gap between **psychology and AI**, offering a **scientific and compassionate** approach to understanding children's emotions.
        
        🖍️ *"Sometimes, the strokes of a child’s drawing reveal more than words ever could."*
        """
    )



# Saving canvas image
def save_canvas_image(image_array):
    """
    Convert the canvas drawing to an RGB image with a white background and black strokes.
    """
    # Convert NumPy array to PIL image (RGBA mode)
    image = Image.fromarray((image_array * 255).astype(np.uint8))  
    image = image.convert("L")
    image = ImageOps.invert(image)
    image = image.convert("RGB")  

    # Save final image
    image.save(IMAGE_PATH)
    return IMAGE_PATH



# House Drawing Analysis Page
if selected == "House Drawing Analysis":
    st.title("🏠 House Drawing Analysis")
    st.write("This tool helps analyze children's psychology based on house drawings. Draw a house or upload a picture for analysis.")

    st.markdown("""
    ### 🧠 **How House Drawings Reflect Children's Psychology**
    Children’s house drawings serve as a **mirror of their emotions, security, and social connections.** Research suggests that certain elements in their drawings reveal their emotional well-being:

    - 🏡 **Small, enclosed houses** → May indicate **insecurity** or **feelings of isolation**  
    - 🚪 **Locked doors, no windows** → Could suggest **communication barriers or withdrawal**  
    - 🎨 **Bright colors, open doors, detailed surroundings** → Show **confidence and social adaptability**  
    - 🌑 **Dark shading, broken roofs, isolated structures** → Might reflect **stress or emotional distress**  

    Studies on **Left-Behind Children (LBC)** reveal that their drawings often depict **lonely or incomplete homes,** reflecting emotional struggles due to parental separation. **Analyzing these drawings helps detect stress, anxiety, or social tendencies,** aiding in early psychological support.
    """)


    # Directory where images will be stored
    PREDICT_PATH = "toPredict"
    os.makedirs(PREDICT_PATH, exist_ok=True)

    IMAGE_PATH = os.path.join(PREDICT_PATH, "predictHouse.png")

    # Layout for better organization
    col1, col2 = st.columns([1, 1])  # Two equal columns

    with col1:  # Left side: Drawing Canvas
        st.subheader("🖌️ Draw Your House Here")
        st.write("Use the canvas below to draw a house. Once finished, save it before predicting.")

        canvas_result = st_canvas(
            fill_color="rgba(255,255,255,1)",  # White background
            stroke_width=3,
            stroke_color="black",
            background_color="white",
            height=224,
            width=224,
            drawing_mode="freedraw",
            key="canvas"
        )

        if st.button("💾 Save Drawing"):
            if canvas_result.image_data is not None:
                img_array = np.array(canvas_result.image_data)  # Convert canvas data to NumPy array

                if np.all(img_array[:, :, :3] == 255):  # Check if all pixels are white
                    st.warning("⚠️ Your canvas is empty! Please draw before saving.")
                else:
                    save_canvas_image(img_array)
                    st.success("✅ Drawing saved successfully!")

    with col2:  # Right side: Upload Image Section
        st.subheader("📤 Upload a House Drawing")
        st.write("If you've already drawn a house on paper, you can upload a scanned image here.")

        uploaded_file = st.file_uploader("Choose an image (PNG, JPG, JPEG)...", type=["png", "jpg", "jpeg"])

    # Prediction function
    def predict_image(image_path):
        """
        Pass the saved image to the models and return the majority prediction.
        """
        housePredict = [
            predict("model/house/house_model_10.tar", image_path),
            predict("model/house/house_model_12.tar", image_path),
            predict("model/house/house_model_15.tar", image_path),
        ]

        final_prediction = Counter(housePredict).most_common(1)[0][0]
        return final_prediction

    # Prediction Section
    st.subheader("🔍 Get Your Result")
    st.write("Once you have either drawn a house or uploaded an image, click the 'Predict' button below.")

    if st.button("🔍 Predict"):
        image_path = None

        if uploaded_file is not None:
            # Save uploaded image as the new input
            image = Image.open(uploaded_file)
            image.save(IMAGE_PATH)
            image_path = IMAGE_PATH
        elif os.path.exists(IMAGE_PATH):
            # If no upload, use the saved canvas image (if available)
            image_path = IMAGE_PATH

        # Process and predict
        if image_path:
            col1, col2 = st.columns([1, 2])  # Adjusting proportions for better readability

            with col1:  # Left side: Processed Image
                st.image(image_path, caption="🖼 Processed Image", width=210)  # Reduced size for a cleaner look

            with col2:  # Right side: Prediction Result & Explanation
                result = predict_image(image_path)

                if result == 0:
                    st.info("❌ **Prediction Result: Stress or Anxiety Detected** [0]")
                    st.write(
                        "The drawing suggests signs of **stress, insecurity, or anxiety.** "
                        "Children experiencing these emotions might draw **unstable lines, excessive erasures, small or enclosed houses,** "
                        "or use **darker shading.** A house with barriers (fences, locked doors) may reflect **emotional detachment or fear.** "
                        "Consider observing the child's **overall behavior and emotional well-being** to offer support."
                    )

                elif result == 1:
                    st.info("✅ **Prediction Result: Introverted & Thoughtful Personality** [1]")
                    st.write(
                        "The drawing indicates that the child might be **reserved, introspective, or cautious in social interactions.** "
                        "An introverted child often focuses on **precise details, symmetrical shapes, and enclosed spaces,** "
                        "suggesting a **desire for structure and predictability.** "
                        "A house with **small windows or without a door** can symbolize a tendency to **keep emotions private.** "
                        "Encouraging **creative expression and open communication** can help such children feel more comfortable sharing their thoughts."
                    )

                else:
                    st.info("🟡 **Prediction Result: Balanced & Social Personality** [2]")
                    st.write(
                        "The drawing reflects a **neutral, balanced, and emotionally stable state.** "
                        "Children who are socially confident and expressive may draw **open doors, bright colors, and larger windows,** "
                        "indicating a **welcoming and friendly nature.** "
                        "A well-proportioned house with decorations and surrounding elements (trees, sun, clouds) "
                        "often represents **happiness, curiosity, and a sense of security.** "
                        "Such children may be more **extroverted, adaptable, and open to new experiences.**"
                    )

        else:
            st.warning("⚠️ Please draw an image and save it or upload a file before predicting.")



# Tree Drawing Analysis Page
if selected == 'Tree Drawing Analysis':
    st.title("🌳 Tree Drawing Analysis")
    st.write("This tool helps analyze children's psychology based on tree drawings. Draw a tree or upload a picture for analysis.")

    # Directory where images will be stored
    PREDICT_PATH = "toPredict"
    os.makedirs(PREDICT_PATH, exist_ok=True)

    IMAGE_PATH = os.path.join(PREDICT_PATH, "predictTree.png")

    # Layout for better organization
    col1, col2 = st.columns([1, 1])  # Two equal columns

    with col1:  # Left side: Drawing Canvas
        st.subheader("🖌️ Draw Your Tree Here")
        st.write("Use the canvas below to draw a Tree. Once finished, save it before predicting.")

        canvas_result = st_canvas(
            fill_color="rgba(255,255,255,1)",  # White background
            stroke_width=3,
            stroke_color="black",
            background_color="white",
            height=224,
            width=224,
            drawing_mode="freedraw",
            key="canvas"
        )

        if st.button("💾 Save Drawing"):
            if canvas_result.image_data is not None:
                img_array = np.array(canvas_result.image_data)  # Convert canvas data to NumPy array

                if np.all(img_array[:, :, :3] == 255):  # Check if all pixels are white
                    st.warning("⚠️ Your canvas is empty! Please draw before saving.")
                else:
                    save_canvas_image(img_array)
                    st.success("✅ Drawing saved successfully!")

    with col2:  # Right side: Upload Image Section
        st.subheader("📤 Upload a Tree Drawing")
        st.write("If you've already drawn a Tree on paper, you can upload a scanned image here.")

        uploaded_file = st.file_uploader("Choose an image (PNG, JPG, JPEG)...", type=["png", "jpg", "jpeg"])

    # Prediction function
    def predict_image(image_path):
        """
        Pass the saved image to the models and return the majority prediction.
        """
        treePredict = [
            predict("model/tree/tree_model_10.tar", image_path),
            predict("model/tree/tree_model_12.tar", image_path),
            predict("model/tree/tree_model_15.tar", image_path),
        ]

        final_prediction = Counter(treePredict).most_common(1)[0][0]
        return final_prediction

    # Prediction Section
    st.subheader("🔍 Get Your Result")
    st.write("Once you have either drawn a Tree or uploaded an image, click the 'Predict' button below.")

    if st.button("🔍 Predict"):
        image_path = None

        if uploaded_file is not None:
            # Save uploaded image as the new input
            image = Image.open(uploaded_file)
            image.save(IMAGE_PATH)
            image_path = IMAGE_PATH
        elif os.path.exists(IMAGE_PATH):
            # If no upload, use the saved canvas image (if available)
            image_path = IMAGE_PATH

        # Process and predict
        if image_path:
            col1, col2 = st.columns([1, 2])  # Adjusting proportions for better readability

            with col1:  # Left side: Processed Image
                st.image(image_path, caption="🖼 Processed Image", width=210)  # Reduced size for a cleaner look

            with col2:  # Right side: Prediction Result & Explanation
                result = predict_image(image_path)

                if result == 0:
                    st.info("❌ **Prediction Result: Emotional Distress Detected** [0]")
                    st.write(
                        "The drawing suggests signs of **stress, anxiety, or emotional insecurity.** "
                        "Children experiencing distress may draw trees with **bare branches, broken limbs, or an overall weak trunk structure.** "
                        "A tree with **thin or unstable roots** might indicate a **lack of security** in their environment. "
                        "Heavy shading, dark colors, or excessive erasing can reflect **internal emotional struggles.** "
                        "Consider providing the child with emotional support and observing their behavior for signs of distress."
                    )

                elif result == 1:
                    st.info("🟡 **Prediction Result: Reserved or Introspective Nature** [1]")
                    st.write(
                        "The drawing suggests that the child is **reserved, introspective, or prefers solitude.** "
                        "An introverted child may draw trees with **symmetrical branches, small leaves, or detailed patterns** "
                        "that show a focus on precision and internal reflection. "
                        "A tree with **a narrow trunk and minimal surrounding elements** may suggest a desire for personal space and structure. "
                        "Encouraging **creative expression** and fostering **gentle social interactions** can help such children feel more engaged."
                    )

                else:
                    st.info("✅ **Prediction Result: Confident & Socially Expressive Personality** [2]")
                    st.write(
                        "The drawing suggests **confidence, emotional balance, and a socially expressive nature.** "
                        "Children who are extroverted and well-adjusted may draw trees with **thick trunks, full leafy canopies, and additional elements** "
                        "like **birds, fruit, or a bright sun,** indicating **curiosity, creativity, and a sense of security.** "
                        "A well-grounded tree with visible **roots and strong branches** can reflect a stable and outgoing personality. "
                        "These children tend to be **open to experiences, enjoy social interactions, and express themselves freely.**"
                    )


        else:
            st.warning("⚠️ Please draw an image and save it or upload a file before predicting.")




# Person Drawing Analysis Page
if selected == 'Person Drawing Analysis':
    st.title("🧑🏻‍🦱 Person Drawing Analysis")
    st.write("This tool helps analyze children's psychology based on person drawings. Draw a person or upload a picture for analysis.")

    # Directory where images will be stored
    PREDICT_PATH = "toPredict"
    os.makedirs(PREDICT_PATH, exist_ok=True)

    IMAGE_PATH = os.path.join(PREDICT_PATH, "predictPerson.png")

    # Layout for better organization
    col1, col2 = st.columns([1, 1])  # Two equal columns

    with col1:  # Left side: Drawing Canvas
        st.subheader("🖌️ Draw Your Person Here")
        st.write("Use the canvas below to draw a person. Once finished, save it before predicting.")

        canvas_result = st_canvas(
            fill_color="rgba(255,255,255,1)",  # White background
            stroke_width=3,
            stroke_color="black",
            background_color="white",
            height=224,
            width=224,
            drawing_mode="freedraw",
            key="canvas"
        )

        if st.button("💾 Save Drawing"):
            if canvas_result.image_data is not None:
                img_array = np.array(canvas_result.image_data)  # Convert canvas data to NumPy array

                if np.all(img_array[:, :, :3] == 255):  # Check if all pixels are white
                    st.warning("⚠️ Your canvas is empty! Please draw before saving.")
                else:
                    save_canvas_image(img_array)
                    st.success("✅ Drawing saved successfully!")

    with col2:  # Right side: Upload Image Section
        st.subheader("📤 Upload a person drawing")
        st.write("If you've already drawn a person on paper, you can upload a scanned image here.")

        uploaded_file = st.file_uploader("Choose an image (PNG, JPG, JPEG)...", type=["png", "jpg", "jpeg"])

    # Prediction function
    def predict_image(image_path):
        """
        Pass the saved image to the models and return the majority prediction.
        """
        personPredict = [
            predict("model/person/person_model_10.tar", image_path),
            predict("model/person/person_model_12.tar", image_path),
            predict("model/person/person_model_15.tar", image_path),
        ]

        final_prediction = Counter(personPredict).most_common(1)[0][0]
        return final_prediction

    # Prediction Section
    st.subheader("🔍 Get Your Result")
    st.write("Once you have either drawn a person or uploaded an image, click the 'Predict' button below.")

    if st.button("🔍 Predict"):
        image_path = None

        if uploaded_file is not None:
            # Save uploaded image as the new input
            image = Image.open(uploaded_file)
            image.save(IMAGE_PATH)
            image_path = IMAGE_PATH
        elif os.path.exists(IMAGE_PATH):
            # If no upload, use the saved canvas image (if available)
            image_path = IMAGE_PATH

        # Process and predict
        if image_path:
            col1, col2 = st.columns([1, 2])  # Adjusting proportions for better readability

            with col1:  # Left side: Processed Image
                st.image(image_path, caption="🖼 Processed Image", width=210)  # Reduced size for a cleaner look

            with col2:  # Right side: Prediction Result & Explanation
                result = predict_image(image_path)

                if result == 0:
                    st.info("❌ **Prediction Result: Depression Indicators** [0]")
                    st.write(
                        "The drawing suggests possible signs of **depression, sadness, or emotional withdrawal.** "
                        "Children experiencing these emotions may draw trees with **bare branches, broken limbs, or an overall weak trunk.** "
                        "A tree with **thin roots or no grounding** might symbolize feelings of **instability, loneliness, or insecurity.** "
                        "Heavy shading, dark colors, or excessive erasures can reflect **internal emotional struggles.** "
                        "Observing the child’s **behavior, mood changes, and interactions** can help determine whether additional emotional support is needed."
                    )

                elif result == 1:
                    st.info("🟡 **Prediction Result: Social Withdrawal & Reserved Nature** [1]")
                    st.write(
                        "The drawing suggests a tendency towards **social withdrawal, introversion, or emotional detachment.** "
                        "Children with such tendencies often draw trees with **small, controlled branches, minimal leaves, or isolated placement** on the page. "
                        "A tree with **narrow trunks, few details, or distant elements** may reflect a preference for solitude and a reluctance to engage socially. "
                        "Encouraging **open communication, creative expression, and positive social interactions** can help these children feel more secure and engaged."
                    )

                else:
                    st.info("✅ **Prediction Result: Perfectionist & Obsessive Tendencies** [2]")
                    st.write(
                        "The drawing suggests a **strong need for control, precision, or perfectionism.** "
                        "Children displaying obsessive tendencies may draw trees with **excessive symmetry, over-detailed branches, or repetitive patterns.** "
                        "A tree with **rigid structure, unnatural perfection, or overly structured elements** may indicate a focus on control and order. "
                        "While this can reflect **high cognitive engagement and attention to detail,** it may also suggest underlying anxiety or a need for predictability. "
                        "Encouraging **flexibility, playfulness, and emotional expression** can help balance structured thinking with creativity."
                    )



        else:
            st.warning("⚠️ Please draw an image and save it or upload a file before predicting.")