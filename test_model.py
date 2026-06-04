from tensorflow.keras.models import load_model

print("Loading model...")

model = load_model("my_medicinal_plant_model.h5", compile=False)

print("SUCCESS!")
print(model.summary())