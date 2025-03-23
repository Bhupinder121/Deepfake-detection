# import timm
# import torch
# import torch.nn as nn
import cv2
# from torchvision import transforms 
import time

# Open the default camera
cam = cv2.VideoCapture('udp://@127.0.0.1:1232')

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
# ret, frame = cam.read()

# cv2.imwrite(frame, 'frame.jpg')


"""
# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Xception model (without automatic pretraining)
model = timm.create_model("xception", pretrained=False)

# Load manually downloaded model weights
model_path = "./xception_weights2.pth"

num_features = model.fc.in_features  # ✅ Change from `classifier` to `fc`

# Replace the FC layer with a binary classification head
model.fc = nn.Linear(num_features, 1) 

model.load_state_dict(torch.load(model_path, map_location=device))

# Get the number of input features for the final FC layer
 # ✅ Replace the last layer for binary classification

# Move model to device
model = model.to(device)
model.eval()

# Define preprocessing transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize for 3 channels
])

t = 0
"""
while True:
    # toogle = "frame1.jpg" if t % 2 == 0 else 'frame2.jpg'
    ret, frame = cam.read()

    # Convert BGR (OpenCV) to RGB
    # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Apply transformations
    # img_tensor = transform(frame).unsqueeze(0).to(device)  # Add batch dimension

    # # Run inference
    # with torch.no_grad():
    #     output = model(img_tensor).squeeze(1)
    #     prediction = torch.sigmoid(output).item()  # Convert logits to probability

    # # Display prediction
    # label = "Real" if prediction > 0.5 else "Fake"
    # cv2.putText(frame, f"{label} ({prediction:.2f})", (10, 50),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Camera', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
cam.release()
cv2.destroyAllWindows()


