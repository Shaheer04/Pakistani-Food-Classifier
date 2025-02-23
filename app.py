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
    st.subheader(f"🍴 {data['name']}")
    st.caption(f"Serving Size: {data['servingSize']}")
    
    # Create columns layout
    col1, col2, col3 = st.columns(3)
    
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
            delta=data['nutrients']['Total Fat']['dailyValue'],
            delta_color="inverse",
            help="Daily Value percentage"
        )
        
    with col3:
        st.metric(
            label="Protein",
            value=data['nutrients']['Protein']['amount'],
            delta=data['nutrients']['Protein'].get('dailyValue', 'N/A'),
            help="Protein content"
        )
    
    # Second row of metrics
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.metric(
            label="Carbohydrates",
            value=data['nutrients']['Total Carbohydrates']['amount'],
            delta=data['nutrients']['Total Carbohydrates']['dailyValue'],
            delta_color="inverse"
        )
        
    with col5:
        st.metric(
            label="Sugars",
            value=data['nutrients']['Sugars']['amount'],
            delta=data['nutrients']['Sugars'].get('dailyValue', 'N/A')
        )
        
    with col6:
        st.metric(
            label="Dietary Fiber",
            value=data['nutrients']['Dietary Fiber']['amount'],
            delta=data['nutrients']['Dietary Fiber']['dailyValue'],
            delta_color="off"
        )
    
    # Third row for vitamins/minerals
    st.subheader("Vitamins & Minerals")
    col7, col8, col9 = st.columns(3)
    
    with col7:
        st.metric("Sodium", 
                 data['nutrients']['Sodium']['amount'],
                 data['nutrients']['Sodium']['dailyValue'],
                 delta_color="inverse")
        
    with col8:
        st.metric("Iron",
                 data['nutrients']['Iron']['amount'],
                 data['nutrients']['Iron']['dailyValue'],
                 delta_color="off")
        
    with col9:
        st.metric("Calcium",
                 data['nutrients']['Calcium']['amount'],
                 data['nutrients']['Calcium']['dailyValue'],
                 delta_color="off")
    
    # Add visual separator
    st.markdown("---")


# Streamlit app
def main():
    st.title("Foodie 🍟🍛")
    col1,col2 = st.columns(2)

    # Load model once
    model = load_model()

    st.sidebar.subheader("Upload Image of Biryani, Fries or BBQ!")

    uploaded_image = st.sidebar.file_uploader("",type=["jpg", "jpeg", "png"])
                
    if uploaded_image is not None:
        with st.spinner('Predicting...'):
            image = Image.open(uploaded_image)
            st.sidebar.image(image,caption="Uploaded image", width=300) 


        with st.spinner('Analyzing Image...'):

            # Preprocess and predict
            input_tensor = preprocess_image(image)
            predicted_class, confidence = predict_image(input_tensor, model)

            st.success(f"Predicted Food Name: **{predicted_class}**")
            st.progress(confidence, text=f"Confidence Score {confidence:.1%}")

            with st.spinner("Searching database..."):
                results = get_nutrition_data(predicted_class)
                
                if not results:
                    st.error("Food not found in database!")
                else:
                    if results[0]['match_score'] > 1.5:
                        display_nutrition(results[0])
                    else:
                        st.info(f"Found {len(results)} matches")
                        selected = st.selectbox(
                            "Select the best match:",
                            options=results,
                            format_func=lambda x: f"{x['name']}"
                        )
                        display_nutrition(selected)
               
            
if __name__ == '__main__':
    food = main()