# treading-bot

## Chalana (local)

```bash
pip install -r requirements.txt
python app.py            # 0.0.0.0:8000 par chalega
```

Same WiFi par phone se kholna: `http://<laptop-ka-IP>:8000`
(`uvicorn app:app` bina `--host 0.0.0.0` ke sirf `127.0.0.1` par sunta hai — phone se nahi khulega.)

## Deploy (Render / Railway / Heroku jaisa)

- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT` (Procfile me already hai)
