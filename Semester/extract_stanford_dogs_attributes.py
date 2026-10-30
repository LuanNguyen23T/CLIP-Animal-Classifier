import os
import json
import time
import urllib.request
import urllib.error

CLASSES_PATH = "Semester/cached_features/stanford_dogs/classes.json"
OUTPUT_FILE = "Semester/attributes/stanford_dogs_attributes.json"
os.makedirs("Semester/attributes", exist_ok=True)

def load_classes():
    with open(CLASSES_PATH, "r") as f:
        return json.load(f)

def build_prompt(breed_name):
    return f"""You are a professional canine morphologist and veterinarian.
Directly output a valid JSON object describing the fine-grained visual characteristics of the dog breed: "{breed_name}".
Only provide visual characteristics visible in a clear photograph grounded in AKC/FCI standards.
Do NOT output reasoning. Output ONLY valid JSON:
{{
  "breed": "{breed_name}",
  "human_name": "{breed_name}",
  "species": "dog",
  "attributes": {{
    "head_and_ears": "specific shape of skull, muzzle, and ear carriage",
    "eyes": "distinctive eye color and eye shape",
    "coat": "coat texture, fur length, color patterns, or markings",
    "body": "overall body build, tail characteristics, stance",
    "distinguishing_mark": "unique visual trait separating it from similar dog breeds"
  }}
}}"""

def query_gemma_lm_studio(prompt, max_retries=3):
    url = "http://127.0.0.1:1234/v1/chat/completions"
    payload = {
        "model": "google/gemma-4-e4b",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 1200,
        "stream": False
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"}
    )
    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                content = res["choices"][0]["message"].get("content", "").strip()
                if "```" in content:
                    parts = content.split("```")
                    for p in parts:
                        p = p.strip()
                        if p.startswith("json"):
                            p = p[4:].strip()
                        if p.startswith("{") and p.endswith("}"):
                            content = p
                            break
                elif "{" in content and "}" in content:
                    start = content.find("{")
                    end = content.rfind("}") + 1
                    content = content[start:end]
                parsed = json.loads(content)
                if "attributes" in parsed and all(k in parsed["attributes"] for k in ["head_and_ears", "eyes", "coat", "body", "distinguishing_mark"]):
                    return parsed
                else:
                    raise ValueError(f"Missing attributes in response: {parsed}")
        except Exception as e:
            if attempt == max_retries:
                raise e
            time.sleep(1.5)

def main():
    classes = load_classes()
    attributes_bank = {}
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r") as f:
                attributes_bank = json.load(f)
            print(f"🔄 Đã tìm thấy {len(attributes_bank)}/{len(classes)} giống đã trích xuất trước đó.")
        except Exception:
            attributes_bank = {}

    print("=" * 70)
    print("🐾 BẮT ĐẦU TRÍCH XUẤT THUỘC TÍNH 120 GIỐNG CHÓ STANFORD BẰNG GEMMA 4 (LM STUDIO)")
    print(f"Tổng số giống: {len(classes)}")
    print("=" * 70)

    t_start = time.time()
    for i, breed in enumerate(classes, 1):
        if breed in attributes_bank and "attributes" in attributes_bank[breed]:
            continue

        print(f"[{i:03d}/{len(classes)}] Đang xử lý: '{breed}'...", end="", flush=True)
        t0 = time.time()
        try:
            prompt = build_prompt(breed)
            res = query_gemma_lm_studio(prompt)
            attributes_bank[breed] = res
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(attributes_bank, f, indent=2, ensure_ascii=False)
            dt = time.time() - t0
            print(f" ✅ ({dt:.1f}s)")
        except Exception as e:
            print(f" ❌ LỖI: {e}")

    total_time = time.time() - t_start
    print("=" * 70)
    print(f"🎉 HOÀN THÀNH TOÀN BỘ {len(attributes_bank)}/{len(classes)} GIỐNG TRONG {total_time:.1f}s!")
    print(f"Đã lưu tại: {OUTPUT_FILE}")
    print("=" * 70)

if __name__ == "__main__":
    main()
