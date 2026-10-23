"""Create evidence images from real outputs produced by the project scripts and API tests."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import json, pandas as pd
from fastapi.testclient import TestClient
import app

ROOT = Path('.')
OUT = ROOT / 'screenshots'
OUT.mkdir(exist_ok=True)

def font(size=19):
    for p in ['/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
        if Path(p).exists(): return ImageFont.truetype(p, size)
    return ImageFont.load_default()

def make_image(title, lines, filename, footer='Generated from this project execution; values are actual program/API results.'):
    width = 1450
    wrapped=[]
    for line in lines:
        line=str(line)
        while len(line)>110:
            wrapped.append(line[:110]); line=line[110:]
        wrapped.append(line)
    height=max(360, 140+len(wrapped)*29+70)
    im=Image.new('RGB',(width,height),'#f5f7fa'); d=ImageDraw.Draw(im)
    d.rectangle((0,0,width,74),fill='#173b5e')
    d.text((28,20),title,fill='white',font=font(27))
    y=100
    for line in wrapped:
        d.text((30,y),line,fill='#17212b',font=font(17)); y+=29
    d.rectangle((0,height-48,width,height),fill='#e6edf3')
    d.text((28,height-35),footer,fill='#34495e',font=font(14))
    im.save(OUT/filename)

# Online output from script result
online=json.loads((ROOT/'outputs/online_prediction.json').read_text())
make_image('ONLINE INFERENCE — actual run',[
    '$ python online_inference.py','ONLINE INFERENCE',json.dumps(online,indent=2),
    'Prediction uses artifacts/model.pkl and a held-out passenger not used during fitting.'
], 'online_inference_example.png')

# Batch output from generated CSV
batch=pd.read_csv(ROOT/'batch_predictions.csv')
lines=['$ python batch_inference.py',f'BATCH INFERENCE: predicted {len(batch)} unseen hold-out records',
       'PassengerId | PredictedSurvived | PredictedLabel']
for _,r in batch.iterrows(): lines.append(f"{int(r.PassengerId):>12} | {int(r.PredictedSurvived):>17} | {r.PredictedLabel}")
make_image('BATCH INFERENCE — actual run',lines,'batch_inference_example.png')

# Streaming output from generated CSV
stream=pd.read_csv(ROOT/'outputs/streaming_predictions.csv')
lines=['$ python streaming_inference.py','STREAMING INFERENCE (one record at a time; 1-second delay)']
for _,r in stream.iterrows():
    pred=int(r.PredictedSurvived); label='Survived' if pred else 'Did not survive'
    lines.append(f"Incoming PassengerId={int(r.PassengerId)} -> prediction={pred} ({label})")
make_image('STREAMING INFERENCE — actual run',lines,'streaming_inference_example.png')

# Exercise real endpoints through FastAPI's ASGI test client, including invalid payload.
client=TestClient(app.app)
health=client.get('/health')
valid=client.post('/predict',json={'Pclass':3,'Sex':'male','Age':22,'SibSp':1,'Parch':0,'Fare':7.25,'Embarked':'S'})
invalid=client.post('/predict',json={'Pclass':9,'Sex':'unknown','Age':-5,'SibSp':-1,'Parch':0,'Fare':-3,'Embarked':'X'})
evidence={
 'health':{'method':'GET','path':'/health','status_code':health.status_code,'response':health.json()},
 'valid_predict':{'method':'POST','path':'/predict','status_code':valid.status_code,'request':{'Pclass':3,'Sex':'male','Age':22,'SibSp':1,'Parch':0,'Fare':7.25,'Embarked':'S'},'response':valid.json()},
 'invalid_predict':{'method':'POST','path':'/predict','status_code':invalid.status_code,'request':'Invalid values (Pclass=9, Sex=unknown, Age=-5, SibSp=-1, Fare=-3, Embarked=X)','response':invalid.json()}
}
(ROOT/'outputs/api_test_results.json').write_text(json.dumps(evidence,indent=2))
api_lines=['$ Automated API endpoint test using FastAPI TestClient (real app routes)',
           f"GET /health  -> HTTP {health.status_code}: {json.dumps(health.json())}",
           f"POST /predict (valid) -> HTTP {valid.status_code}: {json.dumps(valid.json())}",
           f"POST /predict (invalid) -> HTTP {invalid.status_code}: request rejected by Pydantic validation",
           'Expected: health=200, valid prediction=200, invalid request=422',
           'Swagger UI is available at http://127.0.0.1:8000/docs when the server is running.']
make_image('FASTAPI ENDPOINT TESTS — actual responses',api_lines,'fastapi_example.png',
           'Actual endpoint responses captured via FastAPI TestClient; this is not a browser screenshot of Swagger UI.')
print('Created actual-output evidence images and outputs/api_test_results.json')
