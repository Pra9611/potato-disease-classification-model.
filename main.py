from fastapi import FastAPI, UploadFile, File 
import numpy as np
import uvicorn
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

MODEL = tf.keras.models.load_model("../models/model_v1.keras")
CLASS_NAMES = ["Potato___Early_blight","Potato___Late_blight","Potato___healthy"]

@app.get("/ping")
async def ping():
    return "hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
   image = np.array(Image.open(BytesIO(data)))
   return image
   
@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image,0)
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    print(predicted_class, confidence)
    return{
        "class":predicted_class, 
        "confidence":float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost',port=8000)