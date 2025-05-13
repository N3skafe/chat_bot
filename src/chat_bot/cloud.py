import os
import chromadb
import requests
import json
import time
import gradio as gr
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

class SimpleRAG:
    def __init__(
        self, 
        collection_name: str = "documents", 
        embedding_model_name: str = "all-MiniLM-L6-v2",
        chroma_db_path: str = "./chroma_db",
        ollama_base_url: str = "http://localhost:11434"
    ):
        # 임베딩 모델 초기화
        print(f"임베딩 모델 '{embedding_model_name}' 로딩 중...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # ChromaDB 클라이언트 초기화
        print(f"ChromaDB 초기화 중 (경로: {chroma_db_path})...")
        self.db_path = chroma_db_path
        os.makedirs(self.db_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        # Ollama 기본 URL 설정
        self.ollama_base_url = ollama_base_url
        
        # 컬렉션 가져오기 또는 생성
        try:
            self.collection = self.client.get_collection(collection_name)
            print(f"기존 컬렉션 '{collection_name}'을 불러왔습니다. 문서 수: {self.collection.count()}")
        except Exception as e:
            print(f"새로운 컬렉션 '{collection_name}'을 생성합니다.")
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
    
    def add_documents(
        self,
        documents: List[str],
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 32
    ):
        """문서를 ChromaDB에 추가"""
        # 문서 ID 생성
        if ids is None:
            start_idx = self.collection.count()
            ids = [f"doc_{start_idx + i}" for i in range(len(documents))]
        
        # 메타데이터 생성
        if metadatas is None:
            metadatas = [{"source": "unknown"} for _ in range(len(documents))]
        
        # 배치 단위로 처리
        total_docs = len(documents)
        for i in range(0, total_docs, batch_size):
            end_idx = min(i + batch_size, total_docs)
            batch_texts = documents[i:end_idx]
            batch_ids = ids[i:end_idx]
            batch_metadatas = metadatas[i:end_idx]
            
            # 문서 임베딩 생성
            batch_embeddings = self.embedding_model.encode(batch_texts).tolist()
            
            # 컬렉션에 문서 추가
            self.collection.add(
                documents=batch_texts,
                embeddings=batch_embeddings,
                ids=batch_ids,
                metadatas=batch_metadatas
            )
        
        print(f"총 {total_docs}개의 문서가 추가되었습니다.")
        return f"총 {total_docs}개의 문서가 추가되었습니다."
    
    def query_documents(self, query: str, n_results: int = 3):
        """쿼리에 관련된 문서 검색"""
        # 쿼리 임베딩 생성
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # 컬렉션에서 유사한 문서 검색
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        return results
    
    def query_ollama(self, prompt: str, model: str = "deepseek-r1:latest"):
        """Ollama API를 통해 LLM에 쿼리 전송"""
        url = f"{self.ollama_base_url}/api/generate"
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            response = requests.post(url, json=data, timeout=30)
            response.raise_for_status()
            return response.json()["response"]
        except requests.exceptions.RequestException as e:
            print(f"Ollama API 오류: {e}")
            return f"오류: Ollama API 연결에 실패했습니다. Ollama가 실행 중인지 확인하세요."
    
    def rag_query(self, user_query: str, n_results: int = 3, model: str = "deepseek-r1:latest"):
        """RAG 파이프라인 - 쿼리, 검색, LLM 응답 생성"""
        # 1. 관련 문서 검색
        search_results = self.query_documents(user_query, n_results)
        
        # 검색 결과가 없다면
        if not search_results["documents"][0]:
            prompt = f"질문: {user_query}\n\n관련 정보를 찾을 수 없습니다. 알고 있는 내용에 기반하여 답변해주세요."
        else:
            # 2. 검색 결과 컨텍스트 구성
            context_parts = []
            
            for i, doc in enumerate(search_results["documents"][0]):
                context_parts.append(f"[문서 {i+1}]\n{doc}")
            
            context = "\n\n".join(context_parts)
            
            # 3. LLM 프롬프트 구성
            prompt = f"""다음 정보를 바탕으로 질문에 답변해주세요:

### 참고 정보:
{context}

### 질문:
{user_query}

### 답변:
"""
        
        # 4. Ollama 모델에 쿼리 전송
        response = self.query_ollama(prompt, model)
        
        return response, search_results

    def process_file(self, file_obj, chunk_size=1000, overlap=200):
        """파일에서 텍스트를 추출하고 청크로 분할"""
        try:
            text = file_obj.decode('utf-8')
            chunks = self._split_text_into_chunks(text, chunk_size, overlap)
            
            # 청크를 문서로 추가
            metadatas = [{"source": "uploaded_file", "chunk_id": i} for i in range(len(chunks))]
            result = self.add_documents(chunks, metadatas=metadatas)
            return result
        except Exception as e:
            return f"파일 처리 중 오류 발생: {str(e)}"

    def _split_text_into_chunks(self, text, chunk_size, overlap):
        """텍스트를 겹치는 청크로 분할"""
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + chunk_size, text_len)
            
            # 단어 경계에서 분할
            if end < text_len:
                while end > start and not text[end].isspace():
                    end -= 1
                if end == start:  # 적절한 공백을 찾지 못한 경우
                    end = min(start + chunk_size, text_len)
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # 다음 청크의 시작 위치 (겹침 고려)
            start = max(end - overlap, start + 1)
        
        return chunks

# 샘플 문서 초기화 함수
def init_sample_documents(rag_system):
    if rag_system.collection.count() == 0:
        print("샘플 문서를 추가합니다...")
        sample_documents = [
            "RAG(Retrieval-Augmented Generation)는 LLM의 환각 현상을 줄이고 최신 정보를 제공하기 위한 기술입니다.",
            "벡터 데이터베이스는 임베딩 벡터를 저장하고 효율적으로 검색할 수 있게 해주는 데이터베이스입니다.",
            "Chroma DB는 파이썬에서 쉽게 사용할 수 있는 오픈소스 임베딩 데이터베이스입니다.",
            "임베딩 모델은 텍스트를 고차원 벡터 공간으로 변환하여 의미적 유사성을 계산할 수 있게 합니다.",
            "DeepSeek는 최근 공개된 오픈소스 LLM 모델 중 하나로, 다양한 자연어 처리 작업에 활용될 수 있습니다.",
            "벡터 검색은 쿼리 텍스트와 가장 유사한 문서를 찾기 위해 코사인 유사도와 같은 거리 측정 방법을 사용합니다.",
            "프롬프트 엔지니어링은 LLM에서 더 나은 결과를 얻기 위해 입력 프롬프트를 최적화하는 기술입니다.",
            "Ollama는 로컬 환경에서 다양한 오픈소스 LLM을 쉽게 실행할 수 있게 해주는 도구입니다."
        ]
        
        # 문서 메타데이터
        metadatas = [{"source": "rag_info", "category": "ai"} for _ in range(len(sample_documents))]
        
        # 문서 추가
        rag_system.add_documents(sample_documents, metadatas=metadatas)
        return f"샘플 문서 {len(sample_documents)}개가 추가되었습니다."
    
    return f"기존 컬렉션에 {rag_system.collection.count()}개의 문서가 있습니다."

# Gradio 인터페이스와 연결할 함수들
def ask_question(query, model_name, num_results, rag_system):
    if not query.strip():
        return "질문을 입력해주세요.", ""
    
    start_time = time.time()
    response, search_results = rag_system.rag_query(
        query, 
        n_results=num_results, 
        model=model_name
    )
    end_time = time.time()
    
    # 검색된 문서 정보 추출
    retrieved_docs = ""
    if search_results["documents"][0]:
        for i, (doc, metadata) in enumerate(zip(search_results["documents"][0], search_results["metadatas"][0])):
            source = metadata.get("source", "unknown")
            retrieved_docs += f"📄 문서 {i+1} (출처: {source}):\n{doc}\n\n"
    else:
        retrieved_docs = "관련 문서를 찾지 못했습니다."
    
    # 응답에 처리 시간 추가
    full_response = f"{response}\n\n⏱️ 처리 시간: {end_time - start_time:.2f}초"
    
    return full_response, retrieved_docs

def upload_file(files, chunk_size, chunk_overlap, rag_system):
    if not files:
        return "파일을 선택해주세요."
    
    results = []
    for file in files:
        result = rag_system.process_file(file, int(chunk_size), int(chunk_overlap))
        results.append(f"파일 '{os.path.basename(file.name)}': {result}")
    
    return "\n".join(results)

def add_text_document(text, rag_system):
    if not text.strip():
        return "문서 내용을 입력해주세요."
    
    rag_system.add_documents([text])
    return f"문서가 성공적으로 추가되었습니다. 현재 총 {rag_system.collection.count()}개의 문서가 있습니다."

def clear_chat_history():
    return "", ""

def main():
    # RAG 시스템 초기화
    rag_system = SimpleRAG()
    
    # 샘플 문서 초기화
    init_message = init_sample_documents(rag_system)
    
    # Gradio 인터페이스 설정
    with gr.Blocks(title="RAG 시스템 - ChromaDB + Ollama") as demo:
        gr.Markdown("# 📚 RAG 시스템 (ChromaDB + Ollama + DeepSeek)")
        gr.Markdown(f"### 현재 문서 수: {rag_system.collection.count()}")
        
        with gr.Tabs():
            # 질문 탭
            with gr.TabItem("질문하기"):
                with gr.Row():
                    with gr.Column(scale=2):
                        query_input = gr.Textbox(
                            label="질문",
                            placeholder="질문을 입력하세요...",
                            lines=2
                        )
                        with gr.Row():
                            model_select = gr.Dropdown(
                                label="모델",
                                choices=["deepseek-r1:latest", "llama3:latest", "phi3:latest"],
                                value="deepseek-r1:latest"
                            )
                            num_results = gr.Slider(
                                label="검색 결과 수",
                                minimum=1,
                                maximum=10,
                                step=1,
                                value=3
                            )
                        
                        with gr.Row():
                            ask_button = gr.Button("질문하기")
                            clear_button = gr.Button("대화 지우기")
                    
                with gr.Row():
                    with gr.Column():
                        answer_output = gr.Textbox(label="답변", lines=10)
                    with gr.Column():
                        docs_output = gr.Textbox(label="검색된 문서", lines=10)
            
            # 문서 추가 탭
            with gr.TabItem("문서 추가"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 파일 업로드")
                        file_input = gr.File(label="파일 선택", file_types=[".txt", ".md", ".csv"], file_count="multiple")
                        with gr.Row():
                            chunk_size = gr.Number(label="청크 크기", value=1000)
                            chunk_overlap = gr.Number(label="청크 겹침", value=200)
                        upload_button = gr.Button("파일 업로드")
                    
                    with gr.Column():
                        gr.Markdown("### 텍스트 직접 입력")
                        text_input = gr.Textbox(label="문서 내용", lines=10, placeholder="여기에 문서 내용을 입력하세요...")
                        add_text_button = gr.Button("텍스트 추가")
                
                upload_result = gr.Textbox(label="결과")
        
        # 이벤트 연결
        ask_button.click(
            fn=lambda q, m, n: ask_question(q, m, n, rag_system),
            inputs=[query_input, model_select, num_results],
            outputs=[answer_output, docs_output]
        )
        
        clear_button.click(
            fn=clear_chat_history,
            inputs=[],
            outputs=[answer_output, docs_output]
        )
        
        upload_button.click(
            fn=lambda f, cs, co: upload_file(f, cs, co, rag_system),
            inputs=[file_input, chunk_size, chunk_overlap],
            outputs=upload_result
        )
        
        add_text_button.click(
            fn=lambda t: add_text_document(t, rag_system),
            inputs=[text_input],
            outputs=upload_result
        )
    
    # Gradio 앱 실행
    demo.launch(share=False)

if __name__ == "__main__":
    main()