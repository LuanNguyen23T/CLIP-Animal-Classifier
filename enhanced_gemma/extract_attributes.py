"""
enhanced/extract_attributes.py

Script trích xuất các thuộc tính hình thái thị giác (Fine-grained Visual Primitives)
cho 37 giống vật nuôi của tập dữ liệu Oxford-IIIT Pet bằng mô hình Google Gemma 4 (google/gemma-4-e4b)
chạy cục bộ qua LM Studio Server (http://127.0.0.1:1234/v1).
"""

import json
import os
import time
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
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "pets_attributes_gemma.json")
BACKUP_DIR = os.path.abspath(os.path.join(OUTPUT_DIR, "..", "R_D", "gemma_attribute_prompting", "data"))
BACKUP_FILE = os.path.join(BACKUP_DIR, "pets_attributes_gemma.json")


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


def query_gemma_lm_studio(
    prompt: str,
    model: str = "google/gemma-4-e4b",
    base_url: str = "http://127.0.0.1:1234/v1",
    temperature: float = 0.2,
    max_tokens: int = 1200,
    retries: int = 3
) -> dict:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
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

    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))
                content = result["choices"][0]["message"].get("content", "")
                
                # Trích xuất JSON từ nội dung text nếu có markdown code block
                cleaned = content.strip()
                if "```" in cleaned:
                    parts = cleaned.split("```")
                    for part in parts:
                        sub = part.strip()
                        if sub.startswith("json"):
                            sub = sub[4:].strip()
                        if sub.startswith("{") and sub.endswith("}"):
                            cleaned = sub
                            break
                elif "{" in cleaned and "}" in cleaned:
                    start_idx = cleaned.find("{")
                    end_idx = cleaned.rfind("}") + 1
                    cleaned = cleaned[start_idx:end_idx]

                parsed = json.loads(cleaned)
                if "attributes" in parsed and isinstance(parsed["attributes"], dict):
                    return parsed
                else:
                    raise ValueError(f"JSON không đúng cấu trúc mong đợi: {parsed.keys()}")
        except Exception as e:
            if attempt == retries:
                raise e
            time.sleep(2)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:1234/v1")
    parser.add_argument("--model", type=str, default="google/gemma-4-e4b")
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(BACKUP_DIR, exist_ok=True)

    # Nạp dữ liệu cũ nếu đã chạy một phần để tiếp tục (resume)
    all_attributes = {}
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                all_attributes = json.load(f)
            print(f"🔄 Đã tìm thấy file thuộc tính hiện có ({len(all_attributes)}/{len(PET_BREEDS)} giống đã hoàn thành).")
        except Exception:
            all_attributes = {}

    print("=" * 75)
    print(" BẮT ĐẦU TRÍCH XUẤT THUỘC TÍNH HÌNH THÁI VẬT NUÔI BẰNG GEMMA 4 (LM STUDIO)")
    print(f" Endpoint:  {args.base_url}")
    print(f" Model ID:  {args.model}")
    print(f" Tổng số:   {len(PET_BREEDS)} giống thú cưng")
    print("=" * 75)

    start_time = time.time()
    for idx, breed in enumerate(PET_BREEDS, 1):
        if breed in all_attributes and "attributes" in all_attributes[breed]:
            print(f"[{idx:02d}/{len(PET_BREEDS)}] Đã có sẵn: '{breed}' -> Bỏ qua.")
            continue

        print(f"[{idx:02d}/{len(PET_BREEDS)}] Đang trích xuất thuộc tính cho: '{breed}'...")
        t0 = time.time()
        prompt = build_attribute_prompt(breed)
        try:
            data = query_gemma_lm_studio(
                prompt=prompt,
                model=args.model,
                base_url=args.base_url,
                temperature=args.temperature,
                max_tokens=1200
            )
            all_attributes[breed] = data
            elapsed = time.time() - t0
            print(f"  [OK] Hoàn tất trong {elapsed:.1f}s | Loài: {data.get('species')} | {list(data.get('attributes', {}).keys())}")
            
            # Lưu tạm thời sau mỗi giống thành công
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(all_attributes, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"  [ERROR] Lỗi khi trích xuất '{breed}': {e}")

    # Ghi file kết quả cuối cùng
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_attributes, f, indent=2, ensure_ascii=False)
    with open(BACKUP_FILE, "w", encoding="utf-8") as f:
        json.dump(all_attributes, f, indent=2, ensure_ascii=False)

    total_time = time.time() - start_time
    print("=" * 75)
    print(f" HOÀN TẤT TRÍCH XUẤT 37 GIỐNG TRONG {total_time:.1f}s!")
    print(f" Lưu tại: {OUTPUT_FILE}")
    print(f" Backup:  {BACKUP_FILE}")
    print("=" * 75)


if __name__ == "__main__":
    main()
