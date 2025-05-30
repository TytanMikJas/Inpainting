from src.scripts.etl_process.ETLProcessor import ETLProcessor

if __name__ == "__main__":
    DATASET_ID = "mahmudulhaqueshawon/cat-image"
    DATA_DIR = "data/raw_data"
    SPLIT_DATA_DIR = "data/data_splits"

    etl_processor: ETLProcessor = ETLProcessor(DATASET_ID, DATA_DIR, SPLIT_DATA_DIR)

    print("Starting ETL process...")
    etl_processor.process()
    print("ETL process completed successfully.")
