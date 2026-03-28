"""Quick script to show extracted acronyms"""
import json

# Load document with UTF-8 encoding
with open('./data/parsed/document.json', 'r', encoding='utf-8') as f:
    doc = json.load(f)

print("Document keys:", list(doc.keys()))
print()

# Check if metadata exists
if 'metadata' not in doc:
    print("No metadata field found in document!")
    print("Document structure might be different than expected")
    exit(1)

# Get acronyms
acronyms = doc.get('metadata', {}).get('acronyms', {})

print(f"Total acronyms extracted: {len(acronyms)}\n")

if len(acronyms) == 0:
    print("No acronyms found in metadata!")
    print("Metadata keys:", list(doc.get('metadata', {}).keys()))
    exit(1)

# Sort by frequency
sorted_acronyms = sorted(acronyms.items(), key=lambda x: x[1].get('frequency', 0), reverse=True)

print("Top 20 most frequently used acronyms:")
print("-" * 80)
for acronym, data in sorted_acronyms[:20]:
    print(f"{acronym:8} = {data['definition']:50} ({data.get('frequency', 0):4} uses, page {data.get('first_occurrence_page', 0):3})")

print("\n" + "=" * 80)
print("Key Decision Point & Review acronyms:")
print("=" * 80)
kdp_acronyms = ['TRL', 'KDP', 'SRR', 'PDR', 'CDR', 'SIR', 'ORR', 'FRR', 'SAR', 'TRR', 'PRR', 'MRR']
for acr in kdp_acronyms:
    if acr in acronyms:
        data = acronyms[acr]
        print(f"{acr:8} = {data['definition']:50} ({data.get('frequency', 0):4} uses)")
