python -m venv venv 
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install fastapi uvicorn transformers torch scikit-learn kiwipiepy bareunpy python-dotenv
python main.py


python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
pip install fastapi uvicorn transformers torch scikit-learn kiwipiepy bareunpy python-dotenv
nohup python3 main.py &
