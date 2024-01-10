# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from segment_anything import SamPredictor, sam_model_registry
sam = sam_model_registry["vit_h"](checkpoint="ckp/sam_vit_h_4b8939.pth")
predictor = SamPredictor(sam)
predictor.set_image("")
masks, _, _ = predictor.predict("")