try:
  import unzip_requirements
except ImportError:
  pass
from utils import *

def lambda_handler(event, context):
  model = get_model(event['model_name'], event['input_shape'], event['input_name'])
  return execute_model(event, model)