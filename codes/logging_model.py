import logging
from pythonjsonlogger import jsonlogger

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def record_factory(*args, **kwargs):
    record = logging.getLogRecordFactory()(*args, **kwargs)
    record.custom_field = 'custom_value'
    return record

def custom_log_record_factory(*args, **kwargs):
    record = logging.getLogRecordFactory()(*args, **kwargs)
    record.model_params = {}
    record.temp_results = None
    return record

class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        # log_record['model_params'] = record.model_params
        # log_record['temp_results'] = record.temp_results


def log_model_params(params):
    logger.info('Model parameters', extra={'model_params': params})

def log_temp_results(results):
    logger.info('Temporary results', extra={'temp_results': results})