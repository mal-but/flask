import torch
import asyncio
import sys
import requests
import numpy as np  # numpy 임포트 추가

# Windows 환경에서 이벤트 루프 정책 설정
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

# 로거 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI()

# 모델과 토크나이저 로드
model_name = 'beomi/KcELECTRA-base-v2022'
logger.info(f"Loading model {model_name}")

# 분류용 모델 로드 (num_labels는 실제 클래스 수에 맞게 설정)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3, output_hidden_states=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 모델을 평가 모드로 전환
model.eval()

# CUDA 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")
model.to(device)

# 바른 API REST 엔드포인트와 API 키 설정
BAREUN_API_URL = 'http://localhost:5757/bareun/api/v1/analyze'
BAREUN_API_KEY = 'koba-STTQRVI-EDAUW6Q-XHQWDBQ-C5YQFXA'


# Pydantic 모델 정의
class SentencePair(BaseModel):
    sentence1: str  # 정답 문장
    sentence2: str  # 평가할 문장


class EvaluationResult(BaseModel):
    sentence1: str
    sentence2: str
    similarity: float
    delivery_score: float
    morph_similarity: float
    final_score: float


# 문장 임베딩 추출 함수 (평균 풀링 사용)
def get_sentence_embedding(sentence: str):
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # 모든 히든 스테이트를 가져옵니다.
        last_hidden_state = hidden_states[-1]  # 마지막 히든 스테이트
        # 모든 토큰의 임베딩을 평균하여 문장 임베딩 생성
        sentence_embedding = torch.mean(last_hidden_state, dim=1)
    return sentence_embedding.cpu().numpy()


# 코사인 유사도 계산 함수
def calculate_cosine_similarity(sentence1: str, sentence2: str):
    embedding1 = get_sentence_embedding(sentence1)
    embedding2 = get_sentence_embedding(sentence2)
    # 임베딩 벡터 정규화
    embedding1 = embedding1 / np.linalg.norm(embedding1, axis=1, keepdims=True)
    embedding2 = embedding2 / np.linalg.norm(embedding2, axis=1, keepdims=True)
    similarity = np.dot(embedding1, embedding2.T)
    similarity_score = float(similarity[0][0])
    return similarity_score


# 바른 API를 사용한 형태소 유사도 계산 함수
def get_morphemes_from_response(response_data):
    morphemes_list = []
    for sentence in response_data.get('sentences', []):
        for token in sentence.get('tokens', []):
            morphemes = token.get('morphemes', [])
            for morpheme in morphemes:
                morphemes_list.append((morpheme['text']['content'], morpheme['tag']))
    return morphemes_list


def compare_morphemes(morphemes1, morphemes2):
    min_len = min(len(morphemes1), len(morphemes2))
    matches = 0

    for i in range(min_len):
        morph1, tag1 = morphemes1[i]
        morph2, tag2 = morphemes2[i]
        if morph1 == morph2 and tag1 == tag2:
            matches += 1

    similarity_ratio = matches / max(len(morphemes1), len(morphemes2))
    return similarity_ratio


def get_morph_similarity(sentence1: str, sentence2: str):
    try:
        headers = {
            'api-key': BAREUN_API_KEY,
            'Content-Type': 'application/json'
        }

        # 첫 번째 문장에 대한 요청
        data1 = {
            "document": {
                "content": sentence1,
                "language": "ko-KR"
            },
            "encoding_type": "UTF8"
        }
        response1 = requests.post(BAREUN_API_URL, headers=headers, json=data1)
        morphemes1 = get_morphemes_from_response(response1.json())

        # 두 번째 문장에 대한 요청
        data2 = {
            "document": {
                "content": sentence2,
                "language": "ko-KR"
            },
            "encoding_type": "UTF8"
        }
        response2 = requests.post(BAREUN_API_URL, headers=headers, json=data2)
        morphemes2 = get_morphemes_from_response(response2.json())

        # 두 문장의 형태소 비교
        morph_similarity = compare_morphemes(morphemes1, morphemes2)
        return morph_similarity

    except Exception as e:
        logger.error(f"Error in morph analysis using Bareun API: {e}")
        return 0.0


# 전달력 분류 함수
def evaluate_delivery(sentence: str):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    predicted_label = torch.argmax(logits, dim=1).cpu().item()
    return int(predicted_label)


# 점수 변환 함수 (임계값 조정)
def calculate_final_score(similarity: float, delivery_score: int, morph_similarity: float):
    if similarity >= 0.95:
        return 100.0  # 유사도가 0.95 이상이면 100점 반환

    # 전달력 점수의 최대값 설정
    max_delivery_score = 2
    normalized_delivery = delivery_score / max_delivery_score  # 0 ~ 1 사이로 정규화

    # 가중치 설정 (유사도 90, 전달력 5, 형태소 5)
    weight_similarity = 90
    weight_delivery = 5
    weight_morph = 5

    # 총점 계산
    total_score = (
            (similarity * weight_similarity) +
            (normalized_delivery * weight_delivery) +
            (morph_similarity * weight_morph)
    )
    total_score = min(max(total_score, 0), 100)  # 0 ~ 100 사이로 클리핑
    return round(total_score, 2)


# API 엔드포인트
@app.post("/evaluate_similarity", response_model=EvaluationResult)
async def evaluate_similarity(sentence_pair: SentencePair):
    logger.info(f"Processing sentence1: {sentence_pair.sentence1} and sentence2: {sentence_pair.sentence2}")
    try:
        # 코사인 유사도 평가
        similarity = calculate_cosine_similarity(sentence_pair.sentence1, sentence_pair.sentence2)
        logger.info(f"Calculated similarity: {similarity:.4f}")

        # 전달력 점수 평가
        delivery_score = evaluate_delivery(sentence_pair.sentence2)
        logger.info(f"Calculated delivery_score: {delivery_score}")

        # 형태소 유사도 평가 (기존 코드 사용)
        morph_similarity = get_morph_similarity(sentence_pair.sentence1, sentence_pair.sentence2)
        logger.info(f"Calculated morph_similarity: {morph_similarity:.4f}")

        # 최종 점수 계산
        final_score = calculate_final_score(similarity, delivery_score, morph_similarity)
        logger.info(f"Calculated final_score: {final_score}")

        # 결과 객체 생성
        result = EvaluationResult(
            sentence1=sentence_pair.sentence1,
            sentence2=sentence_pair.sentence2,
            similarity=similarity,
            delivery_score=delivery_score,
            morph_similarity=morph_similarity,
            final_score=final_score
        )
        return result

    except Exception as e:
        logger.error(f"Error occurred while processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=1234)
