import torch

#################################### For Image ####################################
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from sam3.visualization_utils import draw_box_on_image, normalize_bbox, plot_results
# Load the model
model = build_sam3_image_model()
processor = Sam3Processor(model)
# Load an image
image = Image.open("assets/record/env5_0002_rgb.jpg")
inference_state = processor.set_image(image)
# Prompt the model with text
output = processor.set_text_prompt(state=inference_state, prompt="person")

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]   
print(masks.shape)
plot_results(
    image,
    inference_state,
    save_path="outputs/sam3_result.png",
    show=False
)

# plot_results(image, inference_state)