import numpy as np
import pandas as pd
# from tensorflow import keras
import keras
# from tensorflow.keras import layers
from keras import layers

SEED = 42
keras.utils.set_random_seed(SEED)

# ------- 1. Load the dataset built in build_dataset.py -------
expr = pd.read_csv("data/expression_processed.csv", index_col="COSMIC_ID")
responses = pd.read_csv("data/responses.csv")
genes = pd.read_csv("model/genes.txt", header=None)[0].tolist()
drugs = sorted(responses["DRUG_NAME"].unique())


def make_inputs(rows: pd.DataFrame):
    """Build the model's two inputs and the answers for a set of rows."""
    x_genes = expr.loc[rows["COSMIC_ID"], genes].to_numpy(dtype="float32")[..., np.newaxis]
    x_drug = pd.get_dummies(pd.Categorical(rows["DRUG_NAME"], categories=drugs)).to_numpy(dtype="float32")
    y = rows["label"].to_numpy()
    return (x_genes, x_drug), y


train = responses[responses["split"] == "train"]
val = responses[responses["split"] == "val"]
test = responses[responses["split"] == "test"]
x_train, y_train = make_inputs(train)
x_val, y_val = make_inputs(val)
x_test, y_test = make_inputs(test)

# ------- 2. Build the CNN -------
gene_input = keras.Input(shape=(len(genes), 1), name="genes")
drug_input = keras.Input(shape=(len(drugs),), name="drug")

x = layers.Conv1D(32, kernel_size=7, activation="relu", padding="same")(gene_input)
x = layers.MaxPooling1D(pool_size=4)(x)
x = layers.Conv1D(64, kernel_size=5, activation="relu", padding="same", name="last_conv")(x)
x = layers.MaxPooling1D(pool_size=4)(x)
x = layers.Flatten()(x)
x = layers.Dropout(0.5)(x)

x = layers.Concatenate()([x, drug_input])
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.3)(x)
output = layers.Dense(3, activation="softmax", name="response")(x)

model = keras.Model(inputs=[gene_input, drug_input], outputs=output)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# ------- 3. Train -------
stop_early = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)
model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val), 
    epochs=100, 
    batch_size=64, 
    callbacks=[stop_early], 
    verbose=2,
)

# ------- 4. Save and quick check -------
model.save("model/cnn_model.keras")
pd.Series(drugs).to_csv("model/drugs.txt", index=False, header=False)

_, test_acc = model.evaluate(x_test, y_test, verbose=0)
breast = (test["CANCER_TYPE"] == "Breast Carcinoma").to_numpy()
breast_acc = (model.predict(x_test, verbose=0).argmax(axis=1)[breast] == y_test[breast]).mean()
print(f"\nTest accuracy (all cancer types): {test_acc:.3f}")
print(f"Test accuracy (breast only):      {breast_acc:.3f}")
print("For reference, random guessing on 3 balanced classes scores about 0.333")
