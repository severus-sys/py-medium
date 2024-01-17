import os
import uuid

from fastapi import FastAPI, File, UploadFile

from predictor import DepthEstimationModel
from upload import upload_image_to_imgbb

ALLOWED_EXTENSION = {".jpg", ".jpeg", ".png"}
TEMP_FOLDER = "api_images"
os.makedirs(TEMP_FOLDER, exist_ok=True)
app = FastAPI()
depth_estimator = DepthEstimationModel()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        file_ext = os.path.splitext(file.filename)[1]  # uzantı
        if file_ext not in ALLOWED_EXTENSION:
            return {"error": "Upload file must be in JPG , JPEG , or PNG Format"}

        filename_base = str(uuid.uuid4())
        filename = filename_base + file_ext
        destionation_path = os.path.join(TEMP_FOLDER, filename)
        output_path = os.path.join(TEMP_FOLDER, "output" + filename_base + ".png")

        with open(destionation_path, "wb") as image_data:
            image_data.write(file.file.read())

        depth_estimator.calculate_depthmap(destionation_path, output_path)
        response = upload_image_to_imgbb(output_path)
        return response

    except Exception as e:
        return {"Error ": str(e)}
