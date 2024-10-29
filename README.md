## window
python -m venv venv <br>
.\venv\Scripts\activate <br>
python -m pip install --upgrade pip <br>
pip install fastapi uvicorn transformers torch scikit-learn kiwipiepy bareunpy python-dotenv <br>
python main.py
<br><br>

## linux
python3 -m venv venv <br>
source venv/bin/activate <br>
python3 -m pip install --upgrade pip <br>
pip install fastapi uvicorn transformers torch scikit-learn kiwipiepy bareunpy python-dotenv <br>
nohup python3 main.py & 


## 바른 API (linux 기준)
sudo apt update && sudo apt upgrade -y <br>
docker pull bareunai/bareun:latest <br>
docker run -d --restart unless-stopped --name bareun -p 5757:5757 -p 9902:9902 -v ~/bareun/var:/bareun/var bareunai/bareun:latest <br>


## 바른 AI
https://bareun.ai/
