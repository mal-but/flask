import torch
import asyncio
import sys
import requests

# (Windows 전용 asyncio 설정 부분은 리눅스에서는 불필요하므로 삭제하거나 유지해도 무관)
# Windows 환경에서 이벤트 루프 정책 설정
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
from typing import List

# 로거 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI()

# 모델과 토크나이저 로드
model_name = 'beomi/KcELECTRA-base-v2022'
logger.info(f"Loading model {model_name}")

# num_labels를 명시적으로 설정
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,
    output_hidden_states=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 모델을 평가 모드로 전환
model.eval()

# CUDA 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")
model.to(device)

# 바른 API REST 엔드포인트와 API 키 설정
BAREUN_API_URL = ('http://localhost:5757/bareun'
                  '/api/v1/analyze')
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


# 문장 임베딩 추출 함수
def get_sentence_embedding(sentence: str):
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.hidden_states
        last_hidden_state = hidden_states[-1]
        cls_embedding = last_hidden_state[:, 0, :]
    return cls_embedding.cpu().numpy()


# 코사인 유사도 계산 함수
def calculate_cosine_similarity(sentence1: str, sentence2: str):
    embedding1 = get_sentence_embedding(sentence1)
    embedding2 = get_sentence_embedding(sentence2)
    similarity = cosine_similarity(embedding1, embedding2)
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
        return 0


# 전달력 분류 함수
def evaluate_delivery(sentence: str):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    predicted_label = torch.argmax(logits, dim=1).cpu().item()
    return int(predicted_label)


# 점수 변환 함수 (유사도가 95 이상일 경우 그대로 반환)
def calculate_final_score(similarity: float, delivery_score: int, morph_similarity: float):
    if similarity >= 0.95:
        return similarity * 100  # 유사도가 95 이상이면 다른 결과 무시하고 그대로 반환

    # 전달력 점수의 최대값 설정
    max_delivery_score = 2
    normalized_delivery = delivery_score / max_delivery_score  # 0 ~ 1 사이로 정규화

    # 가중치 설정 (유사도 85, 전달력 5, 형태소 10)
    weight_similarity = 85
    weight_delivery = 5
    weight_morph = 10

    # 총점 계산
    total_score = (
            (similarity * weight_similarity) +
            (normalized_delivery * weight_delivery) +
            (morph_similarity * weight_morph)
    )
    total_score = max(total_score, 0)
    return float(round(total_score, 2))


# API 엔드포인트
@app.post("/evaluate_similarity_batch", response_model=List[EvaluationResult])
async def evaluate_similarity_batch(sentence_pairs: List[SentencePair]):
    results = []
    for sentence_pair in sentence_pairs:
        logger.info(f"Processing sentence1: {sentence_pair.sentence1} and sentence2: {sentence_pair.sentence2}")
        try:
            # 코사인 유사도 평가
            similarity = calculate_cosine_similarity(sentence_pair.sentence1, sentence_pair.sentence2)
            logger.info(f"Calculated similarity: {similarity:.4f}")

            # 전달력 점수 평가 (두 번째 문장에 대해서만)
            delivery_score = evaluate_delivery(sentence_pair.sentence2)
            logger.info(f"Calculated delivery_score: {delivery_score}")

            # 형태소 유사도 평가
            morph_similarity = get_morph_similarity(sentence_pair.sentence1, sentence_pair.sentence2)
            logger.info(f"Calculated morph_similarity: {morph_similarity:.4f}")

            # 최종 점수 계산 (유사도가 95 이상일 경우 100점 반환)
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
            results.append(result)

        except Exception as e:
            logger.error(f"Error occurred while processing: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    return results


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=1234)
