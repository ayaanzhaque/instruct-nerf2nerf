from metrics.clip_metrics import ClipSimilarity
from PIL import Image
import torchvision.transforms as transforms
clip_model = ClipSimilarity(name="ViT-B/32")  # Choose the model type
print(clip_model)

image_0 = Image.open("res/orig-frame_00005.jpg").convert("RGB")
image_1 = Image.open("res/bald-frame_00005.jpg").convert("RGB")
text_0 = "“a photograph of a man"
text_1 = "“a photograph of a clown"


transform = transforms.Compose([
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Resize((224, 224)),  
])

image_0 = transform(image_0).unsqueeze(0)  # Add batch dimension
image_1 = transform(image_1).unsqueeze(0)  


sim_0, sim_1, sim_direction, sim_image = clip_model.forward(
    image_0, image_1, text_0, text_1
)

print(f"Similarity between image 0 and text 0: {sim_0.item()}")
print(f"Similarity between image 1 and text 1: {sim_1.item()}")
print(f"Directional similarity: {sim_direction.item()}")
print(f"Image-to-image similarity: {sim_image.item()}")

