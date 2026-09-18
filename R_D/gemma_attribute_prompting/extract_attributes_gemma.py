"""
extract_attributes_gemma.py

Script tự động trích xuất các thuộc tính hình thái thị giác (Fine-grained Visual Primitives)
cho 37 giống vật nuôi của tập dữ liệu Oxford-IIIT Pet bằng mô hình Google Gemma cục bộ qua Ollama API.
"""

import json
import os
import urllib.request
import urllib.error

PET_BREEDS = [
    "abyssinian", "american_bulldog", "american_pit_bull_terrier", "basset_hound",
    "beagle", "bengal", "birman", "bombay", "boxer", "british_shorthair",
    "chihuahua", "egyptian_mau", "english_cocker_spaniel", "english_setter",
    "german_shorthaired", "great_pyrenees", "havanese", "japanese_chin",
    "keeshond", "leonberger", "maine_coon", "miniature_pinscher", "newfoundland",
    "persian", "pomeranian", "pug", "ragdoll", "russian_blue", "saint_bernard",
    "samoyed", "scottish_terrier", "shiba_inu", "siamese", "sphynx",
    "staffordshire_bull_terrier", "wheaten_terrier", "yorkshire_terrier"
]

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "data", "pets_attributes_gemma.json")


def query_gemma_lm_studio(
    prompt: str, 
    model: str = "local-model", 
    base_url: str = "http://localhost:1234/v1",
    temperature: float = 0.2
) -> str:
    """
    Gửi prompt đến LM Studio Local Server (OpenAI-compatible endpoint).
    LM Studio mặc định chạy tại http://localhost:1234/v1/chat/completions.
    """
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a professional zoologist, veterinarian, and animal morphologist. Respond strictly in pure, valid JSON format."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": temperature,
        "max_tokens": 600,
        "stream": False
    }
    
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, 
        data=data, 
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer lm-studio"
        }
    )
    
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            result = json.loads(response.read().decode("utf-8"))
            content = result["choices"][0]["message"]["content"]
            return content.strip()
    except urllib.error.URLError as e:
        raise ConnectionError(
            f"Không thể kết nối đến LM Studio Server tại {url}.\n"
            f"Vui lòng đảm bảo rằng:\n"
            f"  1. Bạn đã mở ứng dụng LM Studio.\n"
            f"  2. Tại tab 'Developer' hoặc 'Local Server' (icon <->), nhấn 'Start Server' (cổng mặc định 1234).\n"
            f"  3. Mô hình Gemma đã được nạp (Load Model) vào bộ nhớ.\n"
            f"Chi tiết lỗi: {e}"
        )


def build_attribute_prompt(breed: str) -> str:
    human_breed = breed.replace("_", " ").title()
    return f"""You are a professional zoologist, veterinarian, and animal morphologist.
Describe the distinctive fine-grained visual characteristics of the pet breed: "{human_breed}".
Only provide visual characteristics visible in a clear photograph.

Return ONLY a valid JSON object matching the following structure without any markdown backticks or commentary:
{{
  "breed": "{breed}",
  "human_name": "{human_breed}",
  "species": "cat or dog",
  "attributes": {{
    "head_and_ears": "specific shape of head, muzzle, and ear type",
    "eyes": "distinctive eye color and eye shape",
    "coat": "coat texture, fur length, color patterns, or markings",
    "body": "overall body build, tail characteristics, stance",
    "distinguishing_mark": "unique visual trait that separates it from easily confused similar breeds"
  }}
}}"""


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Trích xuất thuộc tính hình thái vật nuôi bằng Gemma qua LM Studio")
    parser.add_argument("--base-url", type=str, default="http://localhost:1234/v1", help="Địa chỉ LM Studio API (mặc định: http://localhost:1234/v1)")
    parser.add_argument("--model", type=str, default="local-model", help="Tên model hoặc identifier trong LM Studio (mặc định: local-model)")
    parser.add_argument("--temperature", type=float, default=0.2, help="Nhiệt độ sinh mẫu (mặc định: 0.2)")
    args = parser.parse_args()

    os.makedirs(os.path.join(OUTPUT_DIR, "data"), exist_ok=True)
    all_attributes = {}

    print("=" * 70)
    print(" BẮT ĐẦU TRÍCH XUẤT THUỘC TÍNH HÌNH THÁI VẬT NUÔI BẰNG GEMMA (LM STUDIO)")
    print(f" Endpoint: {args.base_url}")
    print(f" Model ID: {args.model}")
    print(f" Số lượng giống: {len(PET_BREEDS)}")
    print("=" * 70)

    for idx, breed in enumerate(PET_BREEDS, 1):
        print(f"\n[{idx:02d}/{len(PET_BREEDS)}] Đang trích xuất thuộc tính cho giống: '{breed}'...")
        prompt = build_attribute_prompt(breed)
        try:
            raw_response = query_gemma_lm_studio(
                prompt=prompt, 
                model=args.model, 
                base_url=args.base_url,
                temperature=args.temperature
            )
            # Loại bỏ markdown nếu Gemma có chèn ```json ... ```
            cleaned = raw_response.strip()
            if "```" in cleaned:
                # Lấy phần nội dung bên trong khối code json
                parts = cleaned.split("```")
                for part in parts:
                    sub = part.strip()
                    if sub.startswith("json"):
                        sub = sub[4:].strip()
                    if sub.startswith("{") and sub.endswith("}"):
                        cleaned = sub
                        break

            breed_data = json.loads(cleaned)
            all_attributes[breed] = breed_data
            attrs = breed_data.get("attributes", {})
            print(f"  [OK] Đã trích xuất thành công:")
            for k, v in list(attrs.items())[:2]:
                print(f"       - {k}: {v[:65]}...")
        except Exception as e:
            print(f"  [ERROR] Lỗi khi xử lý '{breed}': {e}")
            all_attributes[breed] = {
                "breed": breed,
                "error": str(e)
            }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_attributes, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f" ĐÃ HOÀN TẤT! File JSON thuộc tính được lưu tại:")
    print(f" -> {OUTPUT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
