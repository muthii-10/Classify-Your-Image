import torch
from torchvision import transforms
from PIL import Image
import json
from input_args import get_input_args

# Parse command line arguments
args = get_input_args()

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")

# Load the class mapping
def load_class_names(json_path):
    if json_path:
        with open(json_path, 'r') as f:
            class_names = json.load(f)
        return class_names
    return None

# Process the image
def process_image(image_path):
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return preprocess(image).unsqueeze(0)

# Load the model checkpoint
def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model = checkpoint['model'] 
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    return model

# Prediction function
def predict(image_path, model, top_k=5):
    image = process_image(image_path).to(device)
    
    with torch.no_grad():
        output = model(image)
        probs = torch.exp(output)  # Get probabilities
        top_probs, top_indices = probs.topk(top_k)
        
    return top_probs.to(device).numpy().flatten(), top_indices.to(device).numpy().flatten()

# Main function
if __name__ == "__main__":
    # Load the model and class names
    model = load_model(args.checkpoint)
    class_names = load_class_names(args.category_names) if args.category_names else None

    # Make predictions
    top_probs, top_indices = predict(args.input, model, args.top_k)

    # Display results
    print("Top K Classes and Probabilities:")
    for i in range(len(top_probs)):
        class_id = top_indices[i]
        if class_names:
            print(f"{class_names[str(class_id)]}: {top_probs[i]:.4f}")
        else:
            print(f"Class ID {class_id}: {top_probs[i]:.4f}")