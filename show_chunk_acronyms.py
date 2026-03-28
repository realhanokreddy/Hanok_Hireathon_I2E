"""Show acronym usage in chunks"""
import json

# Load chunks
with open('./data/chunks/chunks.json', 'r', encoding='utf-8') as f:
    chunks = json.load(f)

print(f"Total chunks: {len(chunks)}\n")

# Find chunks with acronyms
chunks_with_acronyms = [c for c in chunks if c.get('metadata', {}).get('acronyms')]

print(f"Chunks containing acronyms: {len(chunks_with_acronyms)}")
print(f"Chunks without acronyms: {len(chunks) - len(chunks_with_acronyms)}\n")

# Collect all acronyms
all_acronyms = {}
for chunk in chunks_with_acronyms:
    acronyms = chunk.get('metadata', {}).get('acronyms', {})
    for acr, definition in acronyms.items():
        if acr not in all_acronyms:
            all_acronyms[acr] = definition

print(f"Unique acronyms found across all chunks: {len(all_acronyms)}\n")

print("Sample chunks with acronyms:")
print("=" * 80)
for chunk in chunks_with_acronyms[:5]:
    acr_list = list(chunk['metadata']['acronyms'].keys())
    print(f"\nChunk {chunk['chunk_id']} (Type: {chunk['chunk_type']})")
    print(f"  Section: {chunk['section_title']}")
    print(f"  Acronyms ({len(acr_list)}): {', '.join(acr_list[:8])}")
    if len(acr_list) > 8:
        print(f"    ... and {len(acr_list) - 8} more")

print("\n" + "=" * 80)
print("All acronyms found in chunks:")
print("=" * 80)
for i, (acr, definition) in enumerate(sorted(all_acronyms.items())[:30]):
    print(f"{acr:10} = {definition}")
    if i == 29 and len(all_acronyms) > 30:
        print(f"... and {len(all_acronyms) - 30} more")
