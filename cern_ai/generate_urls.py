"""Generate the full URL list for all CMS NanoAOD Run1 datasets."""
import urllib.request
import json

RECORDS = [31104, 31105, 31106, 31107, 31108, 31109, 31110, 31111,
           31113, 31114, 31126, 31127, 31128, 31130, 31131, 31132]

all_urls = []
for recid in RECORDS:
    try:
        url = f"https://opendata.cern.ch/api/records/{recid}"
        data = json.load(urllib.request.urlopen(url, timeout=10))
        for idx in data['metadata'].get('_file_indices', []):
            for f in idx.get('files', []):
                uri = f['uri']
                http_uri = uri.replace('root://eospublic.cern.ch//eos/', 'http://opendata.cern.ch/eos/')
                if 'merged' not in http_uri:
                    all_urls.append(http_uri)
        title = data['metadata'].get('title', '')[:50]
        n = sum(len(idx.get('files', [])) for idx in data['metadata'].get('_file_indices', []))
        print(f"  {recid}: {n} files - {title}")
    except Exception as e:
        print(f"  {recid}: FAILED - {e}")

print(f"\nTotal: {len(all_urls)} files")
# Save to file
with open("cern_ai/data/all_urls.txt", "w") as f:
    for url in all_urls:
        f.write(url + "\n")
print("Saved to cern_ai/data/all_urls.txt")
