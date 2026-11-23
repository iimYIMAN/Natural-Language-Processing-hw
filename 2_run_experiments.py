import json
from mlx_lm import load, generate

def main():
    model, tokenizer = load("mlx-community/Qwen2.5-3B-Instruct-4bit")

    def ask_llm(prompt, max_tokens=128):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return generate(model, tokenizer, prompt=text, max_tokens=max_tokens, verbose=False).strip()

    with open("raw_100_ko.json", "r", encoding="utf-8") as f:
        bench = json.load(f)

    results = []
    for i, item in enumerate(bench, 1):
        ko_txt, rel = item["source_ko"], item["relation"]

        # Context-aware trilingual ground-truth alignment
        p_gt = (
            f"Context (Korean): {ko_txt}\n"
            f"Translate subject='{item['slots_ko']['subject']}' and object='{item['slots_ko']['object']}' "
            f"into English and Thai (Arabic numerals).\n"
            f"Return ONLY valid JSON: {{\"en\": {{\"subject\": \"...\", \"object\": \"...\"}}, "
            f"\"th\": {{\"subject\": \"...\", \"object\": \"...\"}}}}"
        )
        try:
            gt_parsed = json.loads(ask_llm(p_gt).replace("```json", "").replace("```", "").strip())
            item["slots_en"], item["slots_th"] = gt_parsed["en"], gt_parsed["th"]
        except Exception:
            item["slots_en"] = item.get("slots_en", item["slots_ko"])
            item["slots_th"] = item.get("slots_th", item["slots_ko"])

        # H1: Direct extraction under EN / KO / TH prompts
        p_en = (
            f"Extract the subject and object for relation '{rel}' from the Korean text. "
            f"Return ONLY valid JSON with keys 'subject' and 'object' in Thai.\nText: {ko_txt}"
        )
        p_ko = (
            f"다음 한국어 문장에서 '{rel}' 관계의 주체(subject)와 대상(object)을 추출하세요. "
            f"결과는 반드시 태국어 값으로 된 유효한 JSON('subject', 'object' 키)만 출력하세요.\n문장: {ko_txt}"
        )
        p_th = (
            f"จงสกัด subject และ object สำหรับความสัมพันธ์ '{rel}' จากข้อความภาษาเกาหลีต่อไปนี้ "
            f"ตอบกลับเป็น JSON ที่ถูกต้องเท่านั้นโดยใช้ค่าเป็นภาษาไทย (คีย์ 'subject' และ 'object')\nข้อความ: {ko_txt}"
        )

        # H2: Two-step Pivot (KO -> EN -> TH) vs. Control-Step (KO -> KO -> TH)
        s1_en = ask_llm(f"Extract subject and object for relation '{rel}' from the Korean text as JSON in English.\nText: {ko_txt}")
        p_pivot = f"Translate the values in this JSON into Thai. Return ONLY valid JSON with keys 'subject' and 'object':\n{s1_en}"

        s1_ko = ask_llm(f"Extract subject and object for relation '{rel}' from the Korean text as JSON in Korean.\nText: {ko_txt}")
        p_ctrl = f"Translate the values in this JSON into Thai. Return ONLY valid JSON with keys 'subject' and 'object':\n{s1_ko}"

        results.append({
            "id": item["id"],
            "domain": item["domain"],
            "gt_th": item["slots_th"],
            "en_direct": ask_llm(p_en),
            "ko_direct": ask_llm(p_ko),
            "th_direct": ask_llm(p_th),
            "en_pivot": ask_llm(p_pivot),
            "ko_control_step": ask_llm(p_ctrl),
        })

        if i % 20 == 0:
            print(f"Processed {i}/{len(bench)} samples...")

    with open("benchmark_100_clean.json", "w", encoding="utf-8") as f:
        json.dump(bench, f, ensure_ascii=False, indent=2)
    with open("experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("Saved benchmark_100_clean.json and experiment_results.json.")

if __name__ == "__main__":
    main()
