try:
  import unzip_requirements
except ImportError:
  pass

from deepspeech import Model, printVersions
import json
import base64
import io
import numpy as np
import scipy
import scipy.io.wavfile

SAMPLE_RATE = 16000
BEAM_WIDTH = 500
N_FEATURES = 26
N_CONTEXT = 9

ds = Model('model/output_graph.pbmm' , N_FEATURES, N_CONTEXT, 'model/alphabet.txt' , BEAM_WIDTH)

def inferHandler(event, context):
    body = json.loads(event['body'])

    content = base64.b64decode(body['content'])

    bytes = io.BytesIO(content)

    samplerate, data = scipy.io.wavfile.read(bytes)

    recognized_text = ds.stt(data, samplerate)
    
    response = {
        "statusCode": 200,
        "body": recognized_text
    }
    
    return response
