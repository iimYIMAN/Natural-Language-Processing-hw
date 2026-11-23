import json
from datasets import load_dataset

def main():
    ds = load_dataset("klue/klue", "re")
    validation_data = ds["validation"]
    label_names = validation_data.features["label"].names

    domain_map = {
        "Person_Bio": {"per:date_of_birth", "per:place_of_birth", "per:origin"},
        "Person_Role": {"per:employee_of", "per:title"},
        "Org_Info": {"org:founded", "org:place_of_headquarters"},
        "Org_People": {"org:top_members/employees", "org:members"},
    }

    buckets = {k: [] for k in domain_map}
    seen = set()

    for item in validation_data:
        rel = label_names[item["label"]]
        subj, obj = item["subject_entity"]["word"], item["object_entity"]["word"]
        sent = item["sentence"]

        if sent in seen or len(subj) <= 1 or len(obj) <= 1:
            continue

        for domain, rel_set in domain_map.items():
            if rel in rel_set and len(buckets[domain]) < 25:
                seen.add(sent)
                buckets[domain].append({
                    "id": f"{domain}_{len(buckets[domain])+1:02d}",
                    "domain": domain,
                    "relation": rel,
                    "source_ko": sent,
                    "slots_ko": {"subject": subj, "object": obj},
                })
                break

    dataset_100 = [row for rows in buckets.values() for row in rows]
    with open("raw_100_ko.json", "w", encoding="utf-8") as f:
        json.dump(dataset_100, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(dataset_100)} samples to raw_100_ko.json.")

if __name__ == "__main__":
    main()
