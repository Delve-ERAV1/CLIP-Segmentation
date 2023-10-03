# S19

# CLIP Segmentation

CLIP Segmentation Project leverages the power of OpenAI's CLIP model combined with a segmentation decoder to perform image segmentation based on textual prompts. Provide an image and a text prompt, and get segmented masks for each prompt.

## Table of Contents

- [Features](#features)
- [Example Prediction](#example-prediction)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Acknowledgements](#acknowledgements)

## Features

- **Textual Prompt Segmentation**: Segment images based on textual prompts.
- **Multiple Prompts**: Support for multiple prompts separated by commas.
- **Interactive UI**: User-friendly interface for easy image uploads and prompt inputs.

## Example Prediction

![image](https://github.com/Delve-ERAV1/S19/assets/11761529/6828d8b6-00a5-42c4-879f-01e11a0a9936)

Above is an example of a prediction made using the CLIP Segmentation Project. The image on the left is the original, and on the right, you can see the segmented masks based on the provided prompts.

## Installation

Clone the repository:
```bash
git clone https://github.com/yourusername/clip-segmentation.git
cd clip-segmentation
```
Install the required packages:

```bash
pip install -r requirements.txt
```

Run the application:
```bash
python app.py
```

Try out the app on Huggingface Spaces at this [link](https://huggingface.co/spaces/Sijuade/CLIPSegmentation)


## Usage

1. Upload an image using the provided interface.
2. Enter your textual prompts separated by commas.
3. Click on "Visualize Segments get the segmented masks.
4. Hover over a class to view the individual segment.

## How It Works

The CLIP Segmentation Project combines the power of a pretrained CLIP model with a segmentation decoder. The CLIP model, developed by OpenAI, understands images paired with natural language. By combining this with a segmentation decoder, we can generate segmented masks for images based on textual prompts, bridging the gap between vision and language in a unique way.


## Acknowledgements

- Thanks to [OpenAI](https://openai.com/) for the CLIP model.
- Inspired by the research paper: [Image Segmentation Using Text and Image Prompts](https://github.com/timojl/clipseg).
