from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from cnnClassifier.pipeline.stage_03_training import ModelTrainingPipeline
from cnnClassifier.pipeline.stage_04_evaluation import ModelEvaluationPipeline

STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f">>>>>>> stage : {STAGE_NAME} started")
    data_ingeston = DataIngestionTrainingPipeline()
    data_ingeston.main()
    logger.info(f">>>>>>> stage : {STAGE_NAME} completed")

except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Prepare base model"
try:
    logger.info(f"*******************")
    logger.info(f">>>>>>> stage : {STAGE_NAME} started")
    prepare_base_model = PrepareBaseModelTrainingPipeline()
    prepare_base_model.main()
    logger.info(f">>>>>>> stage : {STAGE_NAME} completed")

except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = 'Training'
try:
    logger.info(f"*******************")
    logger.info(f">>>>>>> stage : {STAGE_NAME} started")
    obj = ModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>>> stage : {STAGE_NAME} completed")

except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = 'Model-Evaluation'
try:
    logger.info(f">>>>>>> stage : {STAGE_NAME} started")
    obj = ModelEvaluationPipeline()
    obj.main()
    logger.info(f">>>>>>> stage : {STAGE_NAME} completed")

except Exception as e:
    logger.exception(e)
    raise e