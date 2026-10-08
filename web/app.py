"""
Flask API Backend cho hệ thống Image Caption
- POST /api/generate-captions: nhận ảnh, trả về 5 captions
- GET  /: serve landing page
"""

import os
import io
import re
import pickle
import numpy as np
import base64
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
CORS(app)

# Đường dẫn model
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH   = os.path.join(BASE_DIR, "caption_model_v3.keras")
VOCAB_PATH   = os.path.join(BASE_DIR, "vocab_v2.pkl")

# Hằng số model
START, END, PAD, UNK = "startseq", "endseq", "<pad>", "<unk>"

# Load model
print("Đang load model...")

# Load vocab
with open(VOCAB_PATH, "rb") as f:
    vocab_data = pickle.load(f)

w2i        = vocab_data["w2i"]
i2w        = vocab_data["i2w"]
vocab_size = len(w2i)
PAD_ID     = w2i[PAD]
UNK_ID     = w2i[UNK]
START_ID   = w2i[START]
END_ID     = w2i[END]

# Load caption model
def masked_ce(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)       # (B, T)
    mask = tf.cast(tf.not_equal(y_true, PAD_ID), ce.dtype)
    return tf.reduce_sum(ce * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)


caption_model = load_model(MODEL_PATH, custom_objects={"masked_ce": masked_ce})

# Tự động đồng bộ max_length với input shape của model
if hasattr(caption_model.inputs[1], 'shape') and caption_model.inputs[1].shape[1] is not None:
    max_length = int(caption_model.inputs[1].shape[1])
else:
    max_length = vocab_data["max_length"]

# Load InceptionV3 feature extractor
feature_extractor = tf.keras.applications.InceptionV3(
    weights="imagenet", include_top=False, pooling="avg"
)

print("    Model đã load thành công!")
print(f"   Vocab size: {vocab_size}, Max length: {max_length}")


# Ham tien xu ly anh
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((299, 299))
    arr = np.array(img, dtype="float32")
    arr = arr / 127.5 - 1.0  # Chuẩn hóa [-1, 1]
    return np.expand_dims(arr, axis=0)

# Trich xuat dac trung cua anh
def extract_features(image_bytes):
    tensor = preprocess_image(image_bytes)
    feat   = feature_extractor.predict(tensor, verbose=0)
    return feat[0]  # shape (2048,)


# Hàm kiểm tra lặp n-gram
def _repeats_ngram(seq, nxt, n=3):
    if len(seq) < n - 1:
        return False
    gram = tuple(seq[-(n - 1):]) + (nxt,)
    return any(tuple(seq[i:i + n]) == gram for i in range(len(seq) - n + 1))


# Beam search cải tiến (v3): dùng xác suất ở bước cuối thực sự, tránh lặp n-gram
def generate_caption(feat, beam=3, alpha=0.7, no_repeat=3):
    feat   = np.asarray(feat, dtype="float32").reshape(1, -1)
    beams  = [([START_ID], 0.0)]
    finished = []

    for _ in range(max_length):
        seqs  = [s for s, _ in beams]
        X     = pad_sequences(seqs, maxlen=max_length, padding="post", value=PAD_ID)
        out   = np.array(
            caption_model({"image_feat": np.repeat(feat, len(seqs), axis=0), "seq": X},
                          training=False)
        )
        # Lấy xác suất tại bước cuối thực sự của mỗi beam
        probs = np.stack([out[b, len(s) - 1] for b, s in enumerate(seqs)])
        probs[:, [PAD_ID, UNK_ID, START_ID]] = 0.0

        cand = []
        for (seq, score), p in zip(beams, probs):
            for idx in np.argsort(p)[-(beam * 2):][::-1]:
                if p[idx] <= 0 or _repeats_ngram(seq, int(idx), no_repeat):
                    continue
                cand.append((seq + [int(idx)], score + float(np.log(p[idx] + 1e-12))))
        if not cand:
            break
        cand.sort(key=lambda c: c[1], reverse=True)

        beams = []
        for seq, score in cand[:beam]:
            (finished if seq[-1] == END_ID else beams).append((seq, score))
        if not beams or len(finished) >= beam:
            break

    finished = finished or beams
    best = max(finished, key=lambda c: c[1] / (len(c[0]) ** alpha))[0]
    return " ".join(i2w[i] for i in best[1:] if i != END_ID)


def generate_diverse_captions(feat, n=5):
    """
    Sinh n captions đa dạng bằng cách:
    - 1 caption beam=1 (greedy)
    - 2 captions beam=3
    - 2 captions beam=5
    Sau đó thêm biến thể ngẫu nhiên nhỏ vào feature vector
    """
    captions = set()
    results  = []

    # Beam search với các giá trị beam khác nhau
    for beam_size in [1, 2, 3, 4, 5]:
        # Thêm noise nhỏ vào feature để tạo đa dạng
        noise  = np.random.normal(0, 0.02, feat.shape).astype("float32")
        f_noisy = feat + noise
        cap    = generate_caption(f_noisy, beam=beam_size)
        if cap and cap not in captions:
            captions.add(cap)
            results.append(cap)
        if len(results) >= n:
            break

    # Nếu chưa đủ, thêm caption gốc vào
    if len(results) < n:
        cap = generate_caption(feat, beam=3)
        if cap not in captions:
            results.append(cap)

    return results[:n]


# Route
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/generate-captions", methods=["POST"])
def generate_captions():
    """
    Nhận ảnh qua multipart form-data hoặc base64 JSON
    Trả về 5 captions
    """
    try:
        # Lấy ảnh từ request
        if "image" in request.files:
            image_bytes = request.files["image"].read()
        elif request.is_json and "image_base64" in request.json:
            # Hỗ trợ base64
            b64data    = request.json["image_base64"]
            if "," in b64data:
                b64data = b64data.split(",")[1]
            image_bytes = base64.b64decode(b64data)
        else:
            return jsonify({"error": "Không tìm thấy ảnh trong request"}), 400

        # Trích xuất đặc trưng
        feat = extract_features(image_bytes)

        # Sinh đa dạng 5 captions
        np.random.seed(42)
        captions = generate_diverse_captions(feat, n=5)

        # Đảm bảo có ít nhất 1 caption
        if not captions:
            return jsonify({"error": "Không thể sinh caption"}), 500

        # Trả về JSON
        return jsonify({
            "success": True,
            "captions": captions,
            "count": len(captions)
        })

    except Exception as e:
        print(f"Lỗi: {e}")
        return jsonify({"error": str(e)}), 500

# Kiem tra trang thai server
@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "running",
        "model": "caption_model_v3.keras",
        "vocab_size": vocab_size,
        "max_length": max_length
    })


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)
