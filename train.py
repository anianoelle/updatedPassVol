from model import train_and_save_model

if __name__ == "__main__":
    csv_path = "data/expandedDataset.csv"
    model_path = "model/pujPassModel.pkl"
    encoders_path = "model/encoders.pkl"

 
    print("🚀 Training model...")
    train_and_save_model(csv_path, model_path, encoders_path)
    print("✅ Training finished. Model saved to:", model_path)
