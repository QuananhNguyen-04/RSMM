import json
import os
import re
import time 
import google.generativeai as genai
from dotenv import load_dotenv
from google.api_core.exceptions import InvalidArgument

# ----- Cài đặt Thư viện OSS (Mã nguồn mở) -----
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance, VectorParams, PointStruct,
    HnswConfigDiff,
    Filter, FieldCondition, MatchText, MatchValue
)
from sentence_transformers import SentenceTransformer, CrossEncoder

# ----- (Bước 3) Cấu hình Google Gemini -----
# Load environment variables from .env (if present)
load_dotenv()

# Read API key from environment variable `GOOGLE_API_KEY`
# Fallback remains a placeholder to help detect missing config
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY")

llm_model = None
try:
    # Guard: missing or still the placeholder
    if not GOOGLE_API_KEY or "YOUR_GOOGLE_API_KEY" in GOOGLE_API_KEY:
        raise InvalidArgument("Vui lòng dán API Key của bạn vào biến môi trường GOOGLE_API_KEY.")

    genai.configure(api_key=GOOGLE_API_KEY) # type: ignore
    # Sử dụng Gemini variant phù hợp
    llm_model = genai.GenerativeModel('gemini-2.5-flash') # type: ignore
    print("Đã kết nối thành công với Google Gemini.")

except Exception as e:
    print(f"LỖI KẾT NỐI GEMINI: {e}")

# ----- Cấu hình Model OSS -----
embed_model = None
rerank_model = None
EMBEDDING_DIM = None

try:
    print("Đang tải model AI (có thể mất chút thời gian lần đầu)...")
    # Tải model embedding
    embed_model = SentenceTransformer('BAAI/bge-m3', device='cuda' if os.environ.get("CUDA_VISIBLE_DEVICES") else 'cpu')
    EMBEDDING_DIM = embed_model.get_sentence_embedding_dimension()
    
    # Tải model reranker
    rerank_model = CrossEncoder('BAAI/bge-reranker-large', device='cuda' if os.environ.get("CUDA_VISIBLE_DEVICES") else 'cpu')
    print("Tải model hoàn tất.")
except Exception as e:
    print(f"LỖI TẢI MODEL: {e}")

# -----------------------------------------------

class RAGPipelineAdvanced:
    def __init__(self, input_file="input.txt"):
        if not llm_model or not embed_model or not rerank_model or EMBEDDING_DIM is None:
            print("LỖI: Chưa khởi tạo được các Model AI. Dừng hệ thống.")
            self.client = None
            return

        self.input_file = input_file
        
        # Tạo tên Collection dựa trên tên file
        # Ví dụ: file "trans_f1.json" -> collection "db_trans_f1_json"
        
        safe_name = re.sub(r'[^a-zA-Z0-9]', '_', os.path.basename(input_file))
        self.collection_name = f"db_{safe_name}" 
        # --------------------

        self.db_path = "./qdrant_db_storage"
        self.chat_history = [] 
        self.current_context = ""
        
        # 1. Khởi tạo Qdrant Client (On-disk)
        try:
            self.client = QdrantClient(path=self.db_path)
        except Exception as e:
            print(f"Lỗi khởi tạo Qdrant: {e}")
            self.client = None
            return
        print("Tạo database thành công", self.client)
        # 2. Tải dữ liệu thô vào bộ nhớ (để lấy ngữ cảnh & tóm tắt)
        self.kb_memory = self._load_memory_kb()
        if not self.kb_memory: 
            self.client = None
            return
        print("Tải dữ liệu thô thành công")
        # 3. Kiểm tra và xây dựng lại CSDL Vector nếu cần
        self._check_and_build_db()

    def _load_input_data(self):
        try:
            print(self.input_file)
            with open(self.input_file, 'r', encoding='utf-8') as f:
                content = f.read()
            json_start = content.find('[')
            if json_start == -1: return []
            return json.loads(content[json_start:])
        except Exception: return []

    def _load_memory_kb(self):
        data = self._load_input_data()
        if not data: 
            print("Invalid Input Data")
            return None
        kb = {"utterances": {}, "utterance_order": [], "full_text": ""}
        full_text_list = []
        
        # SỬA LỖI: Xử lý trường hợp thiếu key 'speaker'
        for i, utt in enumerate(data):
            utt_id = f"utt_{i}"
            
            # Gán giá trị mặc định nếu thiếu
            speaker = utt.get('speaker', 'Unknown Speaker')
            text = utt.get('text', '')
            start = utt.get('start', 0)
            
            # Cập nhật lại utt với các key đảm bảo tồn tại
            utt_safe = utt.copy()
            utt_safe['speaker'] = speaker
            utt_safe['text'] = text
            utt_safe['start'] = start
            
            kb["utterances"][utt_id] = utt_safe
            kb["utterance_order"].append(utt_id)
            
            # Tạo văn bản đầy đủ cho tính năng tóm tắt
            full_text_list.append(f"{speaker}: {text}")
        
        kb["full_text"] = "\n".join(full_text_list)
        return kb

    def _check_and_build_db(self):
        if self.client is None or self.kb_memory is None or embed_model is None or EMBEDDING_DIM is None: return

        # Logic kiểm tra cache đơn giản hóa (để demo)
        # Trong thực tế nên kiểm tra timestamp file
        if self.client.collection_exists(self.collection_name): # type: ignore
            # (Tuỳ chọn) Nếu bạn muốn chắc chắn DB mới khớp với file mới, 
            # hãy bỏ comment dòng dưới để xóa DB cũ đi xây lại:
            # self.client.delete_collection(self.collection_name)
            return # Đã có DB
        
        print("\n--- Đang xây dựng CSDL Vector lần đầu... ---")
        self.client.create_collection( # type: ignore
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE), # type: ignore
            hnsw_config=HnswConfigDiff(m=16, ef_construct=100) # type: ignore
        )
        
        # Vector hóa và nạp dữ liệu
        docs = []
        ids = []
        payloads = []
        utterance_ids = self.kb_memory["utterance_order"]
        
        for i, uid in enumerate(utterance_ids):
            utt = self.kb_memory["utterances"][uid]
            docs.append(utt['text']) # Chỉ vector hóa nội dung
            ids.append(i)
            payloads.append({
                "text_original": utt['text'],
                "doc_id": uid,
                "speaker": utt['speaker'] # Key này giờ đã an toàn nhờ _load_memory_kb
            })
            
        embeddings = embed_model.encode(docs, show_progress_bar=True)
        
        points = [
            PointStruct(id=ids[i], vector=embeddings[i].tolist(), payload=payloads[i]) 
            for i in range(len(ids))
        ]
        
        # Batch upsert
        batch_size = 100
        for i in range(0, len(points), batch_size):
            self.client.upsert(self.collection_name, points[i:i+batch_size]) # type: ignore
            
        print("Xây dựng DB hoàn tất!")
    # ==========================================
    # HÀM ĐÓNG KẾT NỐI
    # ==========================================
    def close_connection(self):
        """Đóng kết nối Qdrant để giải phóng thư mục DB"""
        if self.client:
            self.client.close()
            print("Đã đóng kết nối CSDL.")
    # ==========================================
    # TÍNH NĂNG 1: TÓM TẮT CUỘC HỌP
    # ==========================================
    def summarize_meeting(self):
        if not llm_model or not self.kb_memory: return "Lỗi: Hệ thống chưa sẵn sàng."

        print("\n--- Đang tổng hợp và tóm tắt cuộc họp... ---")
        full_transcript = self.kb_memory["full_text"]
        
        
        
        prompt = f"""
        Bạn là một trợ lý thư ký chuyên nghiệp. Dưới đây là biên bản ghi lại của một cuộc họp.
        Hãy viết một bản tóm tắt chi tiết bao gồm:
        1. Mục đích chính của cuộc họp.
        2. Các nội dung chính đã thảo luận.
        3. Các quyết định đã được đưa ra hoặc các bước hành động (Action Items) tiếp theo.
        4. Các mốc thời gian (Deadline) nếu có.

        --- TRANSCRIPT ---
        {full_transcript}
        --- END TRANSCRIPT ---
        
        Bản tóm tắt (bằng tiếng Việt):
        
        QUAN TRỌNG: 
        - Chỉ sử dụng thông tin CÓ TRONG TRANSCRIPT ở trên. 
        - TUYỆT ĐỐI KHÔNG thêm thắt thông tin bên ngoài. 
        - Nếu thông tin không có trong transcript (ví dụ: không có deadline), hãy ghi là "Không được đề cập".
        """
        try:
            response = llm_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Lỗi khi tóm tắt: {e}"
    # ==========================================
    # TÍNH NĂNG 2: CHATBOT HỎI ĐÁP (RAG THÔNG MINH)
    # ==========================================
    def rewrite_query(self, user_query):
        """
        Viết lại câu hỏi dựa trên lịch sử chat để đầy đủ ý nghĩa.
        """
        if not self.chat_history or not llm_model:
            return user_query 
            
        # Lấy 3 lượt hội thoại gần nhất
        recent_history = self.chat_history[-3:] 
        history_str = "\n".join([f"User: {h[0]}\nBot: {h[1]}" for h in recent_history])
        
        prompt = f"""
        Dưới đây là lịch sử trò chuyện và một câu hỏi mới.
        Viết lại câu hỏi mới thành một câu hỏi độc lập, đầy đủ ý nghĩa (thêm chủ ngữ, thay thế đại từ 'nó', 'họ'...).
        Giữ nguyên ngôn ngữ. KHÔNG trả lời câu hỏi, chỉ viết lại nó.
        
        --- Lịch sử ---
        {history_str}
        
        Câu hỏi mới: {user_query}
        
        Câu hỏi viết lại:
        """
        
        try:
            response = llm_model.generate_content(prompt)
            rewritten = response.text.strip()
            # In ra để debug (có thể comment lại nếu muốn giao diện sạch hơn)
            # print(f"   [Debug] Query gốc: '{user_query}' -> Query sửa: '{rewritten}'")
            return rewritten
        except:
            return user_query

    def check_context_sufficiency(self, query, context):
        """
        Kiểm tra xem context hiện tại có đủ để trả lời câu hỏi không.
        """
        if not context or not llm_model: return False
        
        prompt = f"""
        Ngữ cảnh hiện tại:
        ---
        {context}
        ---
        
        Câu hỏi: "{query}"
        
        Dựa VÀO CHÍNH XÁC ngữ cảnh trên, liệu có đủ thông tin để trả lời câu hỏi này không?
        Chỉ trả lời duy nhất một từ: "YES" hoặc "NO".
        """
        try:
            response = llm_model.generate_content(prompt)
            return "YES" in response.text.strip().upper()
        except:
            return False

    def retrieve_context(self, query):
        if self.client is None or embed_model is None or rerank_model is None or self.kb_memory is None:
            return []

        # 1. Tìm kiếm Vector
        query_vec = embed_model.encode([query], normalize_embeddings=True)[0].tolist()
        resp = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vec,                 # <-- dùng 'query', không phải 'query_vector'
                # query_filter=query_filter,       # áp dụng filter nếu có
                limit=30,
                with_payload=True
            )

        hits = resp.points or []
        
        
        if not hits: return []
        
        # 2. Rerank
        pairs = []
        for hit in hits:
            if hit.payload:
                pairs.append([query, hit.payload.get('text_original', '')])
        
        if not pairs: return []

        scores = rerank_model.predict(pairs) # type: ignore
        
        # Lấy Top 5
        ranked_hits = sorted(zip(scores, hits), key=lambda x: x[0], reverse=True)[:5]
        
        # 3. Context Augmentation (Lấy 1 trước, 1 sau)
        final_chunks = []
        processed_ids = set()
        
        # Map ID sang index trong list tuần tự
        id_to_index = {uid: i for i, uid in enumerate(self.kb_memory["utterance_order"])}
        
        for score, hit in ranked_hits:
            if not hit.payload: continue
            doc_id = hit.payload.get('doc_id')
            if not doc_id or doc_id in processed_ids: continue
            
            idx = id_to_index.get(doc_id)
            if idx is None: continue
            
            # Cửa sổ [idx-1, idx+1]
            start = max(0, idx - 1)
            end = min(len(self.kb_memory["utterance_order"]), idx + 2)
            
            chunk_text = []
            for i in range(start, end):
                uid = self.kb_memory["utterance_order"][i]
                processed_ids.add(uid)
                u = self.kb_memory["utterances"][uid]
                chunk_text.append(f"[{u['start']}] {u['speaker']}: {u['text']}")
            
            final_chunks.append("\n".join(chunk_text))
            
        return final_chunks



    def chat(self, user_query):
        if not llm_model: return "Lỗi: Model chưa sẵn sàng."

        # --- TỐI ƯU 1: CHỈ REWRITE NẾU CÓ LỊCH SỬ ---
        # Nếu chưa chat câu nào (list rỗng), thì câu hỏi của user là ngữ cảnh đầy đủ rồi.
        # Không cần tốn 1 request để rewrite.
        if not self.chat_history:
            rewritten_query = user_query
            print(f"   [Smart RAG] Câu hỏi đầu tiên, bỏ qua bước Rewrite.")
        else:
            # Chỉ tốn request này khi đã chat > 1 câu
            rewritten_query = self.rewrite_query(user_query)
            print(f"   [Smart RAG] Đã viết lại câu hỏi: {rewritten_query}")
        
        # --- TỐI ƯU 2: LUÔN LUÔN TÌM KIẾM (BỎ CHECK SUFFICIENCY) ---
        # Tìm kiếm vector là local (offline), tốn ít tài nguyên hơn nhiều so với gọi API Gemini
        # nên cứ tìm kiếm luôn, không cần hỏi AI "có cần tìm không".
        
        print(f"   [Smart RAG] Đang tìm kiếm thông tin...")
        context_chunks = self.retrieve_context(rewritten_query)
        
        if not context_chunks:
            final_context_str = "Không tìm thấy thông tin cụ thể trong tài liệu."
        else:
            final_context_str = "\n---\n".join(context_chunks)
            # Cập nhật context hiện tại (nếu sau này cần dùng lại)
            self.current_context = final_context_str 
        
        # Tạo prompt lịch sử
        history_str = "\n".join([f"User: {h[0]}\nBot: {h[1]}" for h in self.chat_history[-3:]])

        prompt = f"""
        Bạn là trợ lý ảo chuyên nghiệp.
        Dựa vào [THÔNG TIN ĐƯỢC CUNG CẤP] dưới đây để trả lời câu hỏi.
        
        QUY TẮC:
        1. Chỉ trả lời dựa trên thông tin được cung cấp.
        2. Nếu thông tin không đủ, hãy nói "Tài liệu cuộc họp không đề cập đến vấn đề này".
        3. Trả lời ngắn gọn, đi thẳng vào vấn đề.

        --- LỊCH SỬ TRÒ CHUYỆN ---
        {history_str}
        
        --- THÔNG TIN ĐƯỢC CUNG CẤP ---
        {final_context_str}
        
        --- CÂU HỎI ---
        User: {user_query}
        (Ngữ cảnh ẩn: {rewritten_query})
        
        Câu trả lời:
        """
        
        try:
            # --- TỐI ƯU 3: CẤU HÌNH GENERATION ---
            # Giảm max_output_tokens để trả lời nhanh hơn nếu cần
            response = llm_model.generate_content(prompt)
            answer = response.text.strip()
            
            # Lưu lịch sử
            self.chat_history.append((user_query, answer))
            return answer
        except Exception as e:
            return f"Lỗi khi tạo câu trả lời: {e}"

# ==========================================
# CHƯƠNG TRÌNH CHÍNH (MAIN MENU)
# ==========================================
def main():
    print("Đang khởi động hệ thống...")
    current_dir = './transcriptions/'
    # current_dir = './'
    # VÒNG LẶP NGOÀI: CHỌN FILE
    while True:
        files = [f for f in os.listdir(current_dir) if f.endswith('.json')]
        if not files:
            print("LỖI: Không tìm thấy file .json nào!")
            return

        print("\n" + "="*40)
        print(" DANH SÁCH CÁC CUỘC HỌP CÓ SẴN ")
        print("="*40)
        for i, f in enumerate(files):
            print(f"{i + 1}. {f}")
        print(f"{len(files) + 1}. ❌ Thoát chương trình") # Thêm tùy chọn thoát ở đây

        selected_file = None
        while True:
            try:
                choice = input(f"\n>> Chọn file (1-{len(files)+1}): ").strip()
                idx = int(choice) - 1
                if idx == len(files): # Chọn thoát
                    print("Tạm biệt!")
                    return
                if 0 <= idx < len(files):
                    selected_file = files[idx]
                    break
                print("Số không hợp lệ.")
            except ValueError:
                print("Vui lòng nhập số.")

        print(f"\n✅ Đang tải dữ liệu từ: {current_dir}{selected_file}")
        
        # Khởi tạo App
        app = RAGPipelineAdvanced(input_file=f"{current_dir}{selected_file}")
        
        if not app.client:
            print(app.client)
            print("\n!!! KHỞI TẠO THẤT BẠI - Quay lại chọn file...")
            continue # Quay lại chọn file khác

        # VÒNG LẶP TRONG: TƯƠNG TÁC VỚI FILE ĐÃ CHỌN
        back_to_menu = False
        while not back_to_menu:
            print(f"\n--- Đang làm việc với: {selected_file} ---")
            print("1. 📝 Tóm tắt nội dung")
            print("2. 💬 Chatbot hỏi đáp")
            print("3. 🔄 Đổi sang file khác (Đóng DB hiện tại)")
            print("4. ❌ Thoát hẳn")
            
            choice = input("\n>> Nhập lựa chọn: ").strip()
            
            if choice == '1':
                print(app.summarize_meeting())
                input("\n[Enter để tiếp tục...]")
                
            elif choice == '2':
                print("\n--- CHAT (Gõ 'exit' để quay lại) ---")
                app.chat_history = [] 
                while True:
                    q = input("Bạn: ")
                    if q.lower() in ['exit', 'quit', 'thoát']: break
                    if not q.strip(): continue
                    print("Bot đang suy nghĩ...", end="\r")
                    ans = app.chat(q)
                    print(" "*20, end="\r") 
                    print(f"Bot: {ans}")

            elif choice == '3':
                # QUAN TRỌNG: Đóng kết nối trước khi break để chọn file mới
                app.close_connection()
                back_to_menu = True # Thoát vòng lặp trong, ra vòng lặp ngoài

            elif choice == '4':
                app.close_connection()
                print("Tạm biệt!")
                return # Thoát hết
            else:
                print("Sai cú pháp.")

if __name__ == "__main__":
    main()
# **Lưu ý quan trọng:** Trong hàm `main()`, mình đã đổi `input_file="input.txt"` thành `input_file="trans_f1_test.json"` để khớp với file lỗi của bạn. Bạn hãy chạy lại code này nhé!