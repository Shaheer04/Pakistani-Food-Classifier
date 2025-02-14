import streamlit as st
import torch
from torchvision import transforms
import torchvision.models as models
from PIL import Image
from torchvision import datasets
from llm import talk_to_llm


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def get_nutrients(predicted_class, compostition_dict):
    
    prompt = f"""from the given data extract the information of {predicted_class} food their nutrients compostion from the given dictionary.
        {compostition_dict}
        """
    extracted_nutrients = talk_to_llm(prompt=prompt)
    return extracted_nutrients


# Streamlit app
def main():
    st.title("Food Classifier 🍟🍛")
    st.write("Upload an image of french fries, BBQ, or biryani")

    # Load model once
    model = load_model()

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', width=300)
        
        # Preprocess and predict
        with st.spinner('Predicting...'):
            # Preprocess
            input_tensor = preprocess_image(image)
            
            predicted_class, confidence = predict_image(input_tensor, model)

        # Display prediction
        st.success(f"Prediction: **{predicted_class}** with {confidence:.1%} confidence")
        



if __name__ == '__main__':
    food = main()