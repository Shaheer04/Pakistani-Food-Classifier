import streamlit as st
import torch
from torchvision import transforms
import torchvision.models as models
from PIL import Image
from torchvision import datasets
from llm import talk_to_llm
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt




# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load environment variables
load_dotenv()

# MongoDB connection
@st.cache_resource
def get_mongo_connection():
    client = MongoClient(os.getenv("MONGO_URI"))
    return client

def get_nutrition_data(food_name):
    client = get_mongo_connection()
    db = client[os.getenv("DB_NAME")]
    collection = db[os.getenv("COLLECTION_NAME")]

    # Text search query using the index
    query = {
        "$text": {
            "$search": food_name,
            "$caseSensitive": False,
            "$diacriticSensitive": False
        }
    }
    
    # Get results sorted by relevance score
    results = list(collection.find(query, {"score": {"$meta": "textScore"}})
                         .sort([("score", {"$meta": "textScore"})])
                         .limit(5))

    # Clean documents and add similarity score
    cleaned = []
    for doc in results:
        doc.pop('_id', None)
        doc.pop('__v', None)
        cleaned.append({
            **doc,
            "match_score": doc.get("score", 0)
        })
    
    return cleaned



# Load your trained model
def load_model():
    model = models.resnet18(pretrained=False)  # Disable pretrained weights
    num_classes = 3  # Same as during training
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes) 
    model.load_state_dict(torch.load('models/model_01.pth', map_location=device))  # Load your saved model state dictionary
    return model

# Load full dataset to get classnames 
full_dataset = datasets.ImageFolder(
    root='data/Fooddataset',
    transform=None  # No transform applied here yet
)
class_names = full_dataset.classes

# Preprocess the image (must match your training pipeline)
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5773, 0.4623, 0.3385], 
                            std=[0.2559, 0.2411, 0.2455])
    ])
    return transform(image).unsqueeze(0).to(device)

def predict_image(input_tensor, model):

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    # Get results
    predicted_idx = torch.argmax(probabilities).item()
    predicted_class = class_names[predicted_idx]
    confidence = probabilities[predicted_idx].item()

    return predicted_class, confidence

def display_nutrition(data):
    # Display header information
    st.header(f"🍴 {data['name']}")
    st.subheader(f"Serving Size: {data['servingSize']}")
    
    # Create columns layout
    col1, col2, col3 = st.columns(3)
    
    with st.expander("View Nutrition Breakdown!"):
    # Calories metric
        col1.metric(
            label="Calories",
            value=data['calories']['amount'],
            help="Total calories per serving"
        )
        
        # Key Nutrients
        with col2:
            st.metric(
                label="Total Fat",
                value=data['nutrients']['Total Fat']['amount'],
                help="Daily Value percentage"
            )
            
        with col3:
            st.metric(
                label="Protein",
                value=data['nutrients']['Protein']['amount'],
                help="Protein content"
            )
        
        # Second row of metrics
        col4, col5, col6 = st.columns(3)
        
        with col4:
            st.metric(
                label="Carbohydrates",
                value=data['nutrients']['Total Carbohydrates']['amount'],
            )
            
        with col5:
            st.metric(
                label="Sugars",
                value=data['nutrients']['Sugars']['amount'],
            )
            
        with col6:
            st.metric(
                label="Dietary Fiber",
                value=data['nutrients']['Dietary Fiber']['amount'],
               
            )
        
        # Third row for vitamins/minerals
        st.subheader("Vitamins & Minerals")
        col7, col8, col9 = st.columns(3)
        
        with col7:
            st.metric("Sodium", 
                    data['nutrients']['Sodium']['amount'],
                    )
            
        with col8:
            st.metric("Iron",
                    data['nutrients']['Iron']['amount'],
                    )
            
        with col9:
            st.metric("Calcium",
                    data['nutrients']['Calcium']['amount'],
                    )
    
    # Add visual separator
    st.markdown("---")


def create_nutrition_pie(data):
    # Extract values
    protein = float(data['nutrients']['Protein']['amount'].replace('g', ''))
    carbs = float(data['nutrients']['Total Carbohydrates']['amount'].replace('g', ''))
    fat = float(data['nutrients']['Total Fat']['amount'].replace('g', ''))
    
    # Calculate percentages
    total = protein + carbs + fat
    sizes = [protein/total*100, carbs/total*100, fat/total*100]
    
    # Create chart
    fig, ax = plt.subplots()
    ax.pie(sizes, 
           labels=['Protein', 'Carbs', 'Fat'],
           colors=['#ff9999','#66b3ff','#99ff99'],
           autopct='%1.1f%%',
           startangle=90)
    ax.axis('equal')
    
    return fig

# Streamlit app
def main():
    st.title("NutriScan")
    st.info("Upload a food image, get its name and nutritional breakdown, and visualize macronutrients with an interactive pie chart, all in one place!")

    # Load model once
    model = load_model()
    
    st.sidebar.caption("Made with 💖 by Shaheer Jamal")

    uploaded_image = st.sidebar.file_uploader("Upload Image of Biryani, Fries or BBQ!",type=["jpg", "jpeg", "png"])
                
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.sidebar.image(image,caption="Uploaded image", width=300) 
        # Preprocess and predict
        input_tensor = preprocess_image(image)
        predicted_class, confidence = predict_image(input_tensor, model)
        with st.spinner('Analyzing Image...'):
            st.caption("Predicted Food Name")
            st.success(f"**{predicted_class}**")
            st.progress(confidence, text=f"Confidence Score {confidence:.1%}")

        with st.spinner("Searching database..."):
            results = get_nutrition_data(predicted_class)
            
            if not results:
                st.error("Food not found in database!")
            else:
                if results:
                    display_nutrition(results[0])
        
                    with st.expander("Visualize Macro Nutrients!"):
                        st.pyplot(create_nutrition_pie(results[0]))
               
            
if __name__ == '__main__':
    food = main()