
import torch
from PIL import Image
from torchvision import transforms
from clipseg import CLIPDensePredT


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Resize((352, 352)),
])


model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
model.eval()
model.load_state_dict(torch.load('weights/rd64-uni.pth', 
                    map_location=torch.device('cpu')), strict=False)


def predict(image, prompts):
    """
    Predict segmentation masks for the given image based on the provided prompts.

    Parameters:
    - image (PIL.Image): The input image.
    - prompts (str): A comma-separated string of prompts.
    - Model (torch.nn): Segmentation Model.

    Returns:
    - tuple: A tuple containing the resized input image and a list of segmentation masks.
    """
    
    img = transform(image).unsqueeze(0)
    
    # Split the prompts string into a list of individual prompts
    prompts = prompts.split(',')
    num_prompts = len(prompts)

    # Ensure no gradient computation during prediction for performance
    with torch.no_grad():
        # Get model predictions for each prompt
        preds = model(img.repeat(len(prompts), 1, 1, 1), prompts)[0]

    # Convert model predictions to segmentation masks
    masks = [torch.sigmoid(preds[i][0]) for i in range(num_prompts)]
    masks = [(m.squeeze(0).numpy(), prompts[i]) for i, m in enumerate(masks)]
    
    # Return the resized input image and the list of segmentation masks
    return (image.resize((352, 352), Image.LANCZOS), masks)

def get_examples():
   examples = [
      ['images/000013.jpg', 'deer, tree, grass'],
      ['images/000002.jpg', 'train, tracks, electric pole, house'],
      ['images/00125.jpg', 'dog, flowers'],
      ['images/000010.jpg', 'horse, man, fence, buildings, hill'],
      ['images/000004.jpg', 'car, truck, building, sky, traffic light, tree, clouds']
   ]
   return(examples)


def get_html():
  html_string = """
      <!DOCTYPE html>
      <html lang="en">

      <head>
          <meta charset="UTF-8">
          <meta name="viewport" content="width=device-width, initial-scale=1.0">
          <title>Multi-Prompt Image Segmentation</title>
          <link href="https://fonts.googleapis.com/css2?family=Roboto+Slab:wght@400;700&display=swap" rel="stylesheet">

          <style>
              /* General styling */
              body {
                  font-family: 'Roboto Slab', serif;
                  margin: 0;
                  padding: 0;
                  background-color: #f4f4f4;
              }

              .app-header {
                  background: linear-gradient(135deg, #4a90e2, #50e3c2);
                  color: #fff;
                  text-align: center;
                  padding: 40px 0;
                  border-radius: 20px;
                  position: relative;
                  overflow: hidden;
                  box-shadow: 0px 10px 20px rgba(0, 0, 0, 0.1);
              }

              /* Ellipse Overlay */
              .app-header::before {
                  content: "";
                  position: absolute;
                  top: -50%;
                  left: -50%;
                  width: 200%;
                  height: 200%;
                  background: rgba(255, 255, 255, 0.1);
                  transform: rotate(45deg);
                  border-radius: 50%;
              }

              /* Floating Shapes */
              .app-header::after {
                  content: "";
                  position: absolute;
                  top: 20%;
                  right: 10%;
                  width: 70px;
                  height: 70px;
                  background: rgba(255, 255, 255, 0.2);
                  border-radius: 50%;
              }

              .floating-shape {
                  content: "";
                  position: absolute;
                  top: 10%;
                  left: 5%;
                  width: 50px;
                  height: 50px;
                  background: rgba(255, 255, 255, 0.2);
                  border-radius: 50%;
              }

              /* Text Styling */
              .app-title {
                  font-size: 28px;
                  margin: 0;
                  font-weight: 700;
                  text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
              }

              .app-description {
                  font-size: 18px;
                  margin-top: 15px;
                  opacity: 0.9;
                  text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.1);
              }

              /* Wavy Bottom */
              .wavy-bottom {
                  position: absolute;
                  bottom: -10px;
                  left: 0;
                  width: 100%;
                  height: 20px;
                  background: #f4f4f4;
                  border-radius: 100% 100% 0 0;
              }
          </style>
      </head>

      <body>

          <!-- App Header -->
          <div class="app-header">
              <h1 class="app-title">Multi-Prompt Image Segmentation</h1>
              <p class="app-description">Upload an image and provide multiple text prompts separated by commas. Get segmented masks for each prompt.</p>
              <div class="floating-shape"></div>
              <div class="wavy-bottom"></div>
          </div>

          <!-- Rest of the app content will go here -->

      </body>

      </html>


  """

  return(html_string)