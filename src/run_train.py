from TrainingManager import TrainingManager

if __name__ == "__main__":
    csv_path = 'kidneyData.csv'
    img_dir = 'kidney_ct_data'
    training_manager = TrainingManager(csv_path, img_dir, epochs=50)
    training_manager.train_model(csv_path, img_dir)
