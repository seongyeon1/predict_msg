import streamlit as st
import pickle
import torch
from PIL import Image

def preprocess_tdata(sentences):
    # TF-IDF 벡터화
    # 저장된 TfidfVectorizer 모델 불러오기
    with open('./model/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    X = vectorizer.transform(sentences)
    return X

def predict_cluster(new_data):
    # 저장된 K-means 모델 불러오기
    kmeans = torch.load('./model/kmeans_final_8')

    # 새로운 데이터 전처리
    new_data_vectorized = preprocess_tdata(new_data)

    # 새로운 데이터의 군집 예측
    predicted_cluster = kmeans.predict(new_data_vectorized)

    return predicted_cluster

# streamlit 페이지 구성
def main():
    st.title("데이터 군집 예측")
    st.write("새로운 데이터의 군집을 예측합니다.")

    # 예측하고자 하는 새로운 데이터 입력 받기
    new_data = st.text_area("새로운 데이터를 입력하세요.")

    # 제출 버튼 클릭 시 예측 결과 출력
    if st.button("예측"):
        # 새로운 데이터의 군집 예측
        predicted_cluster = predict_cluster([new_data])

        # 예측 결과 출력
        st.write("새로운 데이터의 예측 군집:", predicted_cluster)

        # 이미지 경로
        image_path = f"./kmeans_final/cloud_{predicted_cluster[0]}.png"

        # 이미지 표시
        image = Image.open(image_path)
        st.image(image, caption="예측된 군집에 해당하는 이미지")

if __name__ == '__main__':
    main()