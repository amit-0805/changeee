from train_model import train_model
from test_model import test_model

def run_pipeline():
    print("Starting the Change Detection CNN Pipeline...")
    
    # Step 1: Train the Model
    print("\nTraining the model...")
    train_model()
    print("Model training completed successfully!\n")
    
    # Step 2: Test the Model
    print("Testing the model on a pair of test images...")
    test_model()
    print("Testing completed successfully!\n")
    
    print("Pipeline execution finished!")

if __name__ == "__main__":
    run_pipeline()