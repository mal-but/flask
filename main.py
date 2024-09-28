import torch
import asyncio
import sys

# Windows 환경에서 이벤트 루프 정책 설정
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from kiwipiepy import Kiwi
from bareunpy import Tagger
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

# 형태소 분석기와 문장 분리기 설정
logger.info("Initializing taggers")
tagger = Tagger('koba-STTQRVI-EDAUW6Q-XHQWDBQ-C5YQFXA', 'localhost')  
kiwi = Kiwi()

# Pydantic 모델 정의
class SentencePair(BaseModel):
    sentence1: str  # 정답 문장
    sentence2: str  # 평가할 문장

class EvaluationResult(BaseModel):
    sentence1: str
    sentence2: str
    similarity: float
    delivery_score: float
    morph_penalty: int
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

# 형태소 개수 계산 함수
def get_morph_count(sentence: str):
    try:
        morph_num = len(tagger.morphs(sentence))
        return morph_num
    except Exception as e:
        logger.error(f"Error in morph analysis: {e}")
        return 0

# 전달력 분류 함수
def evaluate_delivery(sentence: str):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    predicted_label = torch.argmax(logits, dim=1).cpu().item()
    return int(predicted_label)

# 점수 변환 함수
def calculate_final_score(similarity: float, delivery_score: int, morph_penalty: int):
    # 전달력 점수의 최대값 설정 (예: 2)
    max_delivery_score = 2
    normalized_delivery = delivery_score / max_delivery_score  # 0 ~ 1 사이로 정규화

    # 가중치 설정 (가중치의 합이 100이 되도록)
    weight_similarity = 75
    weight_delivery = 25

    # 총점 계산 (형태소 페널티를 적용하여 감점)
    total_score = (
        (similarity * weight_similarity) +
        (normalized_delivery * weight_delivery) -
        morph_penalty
    )
    # 총점은 최소 0점 이상으로 제한
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

            # 각 문장의 형태소 개수 계산
            morph_num1 = get_morph_count(sentence_pair.sentence1)
            morph_num2 = get_morph_count(sentence_pair.sentence2)
            morph_diff = abs(morph_num1 - morph_num2)
            logger.info(f"Morph counts: reference={morph_num1}, input={morph_num2}")
            logger.info(f"Morph difference: {morph_diff}")

            # 형태소 차이에 따른 감점 계산
            if morph_diff == 0:
                morph_penalty = 0
            elif morph_diff <= 2:
                morph_penalty = 5
            elif morph_diff <= 4:
                morph_penalty = 10
            else:
                morph_penalty = 20
            logger.info(f"Morph penalty: {morph_penalty}")

            # 최종 점수 계산
            final_score = calculate_final_score(similarity, delivery_score, morph_penalty)
            logger.info(f"Calculated final_score: {final_score}")

            # 결과 객체 생성
            result = EvaluationResult(
                sentence1=sentence_pair.sentence1,
                sentence2=sentence_pair.sentence2,
                similarity=similarity,
                delivery_score=delivery_score,
                morph_penalty=morph_penalty,
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
    uvicorn.run(app, host="127.0.0.1", port=1234)
