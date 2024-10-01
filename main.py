from fastapi import FastAPI, UploadFile, File
import uvicorn
from PIL import Image
import numpy as np
import tensorflow as tf

app = FastAPI()

# Load the best model (Nested U-Net or Attention U-Net)
model = tf.keras.models.load_model('best_model.h5')

@app.post("/predict/")
async def predict_mri(file: UploadFile = File(...)):
    image = Image.open(file.file)
    image = np.array(image.resize((256, 256))) / 255.0
    image = np.expand_dims(image, axis=0)
    
    prediction = model.predict(image)
    return {"prediction": prediction.tolist()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
