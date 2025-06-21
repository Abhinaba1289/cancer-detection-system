import h2o
from h2o.automl import H2OAutoML

# Initialize H2O
h2o.init()

# Load the dataset
data = h2o.import_file("wisc.csv")  # Adjust path if needed

# Convert the target column to categorical
data["diagnosis"] = data["diagnosis"].asfactor()

# Set predictors and response column
target = "diagnosis"
predictors = data.columns
predictors.remove(target)
if "id" in predictors:
    predictors.remove("id")  # Remove 'id' if present

# Split the dataset
train, test = data.split_frame(ratios=[0.8], seed=1234)

# Train AutoML model
aml = H2OAutoML(max_models=10, seed=1, project_name="cancer_prediction")
aml.train(x=predictors, y=target, training_frame=train)

# Save the best model
model_path = h2o.save_model(model=aml.leader, path="models", force=True)
print(f"Model saved to: {model_path}")
