import requests
import tarfile
import xml.etree.ElementTree as ET
import os
import json

url = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz"
filename = "NLMCXR_reports.tgz"
extract_dir = "nih_reports_xml"

#delete corrupted file
if os.path.exists(filename):
    print(f"Removing old/corrupted {filename}...")
    os.remove(filename)

#download with browser header
print("Downloading raw XML reports directly from NIH (2MB)...")
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
response = requests.get(url, headers=headers, stream=True)

if response.status_code == 200:
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download complete!")
else:
    print(f"Failed to download. Server returned status code: {response.status_code}")
    exit()

#extract zip
print("Extracting XML files...")
if not os.path.exists(extract_dir):
    os.makedirs(extract_dir)
with tarfile.open(filename, "r:gz") as tar:
    tar.extractall(path=extract_dir)

formatted_data = []

xml_dir = os.path.join(extract_dir, "ecgen-radiology")

for file in os.listdir(xml_dir):
    if file.endswith(".xml"):
        try:
            tree = ET.parse(os.path.join(xml_dir, file))
            root = tree.getroot()
            
            report_text = ""
            for abstract in root.findall(".//AbstractText"):
                if abstract.get('Label') == 'FINDINGS' and abstract.text:
                    report_text += abstract.text + " "
                if abstract.get('Label') == 'IMPRESSION' and abstract.text:
                    report_text += abstract.text
            
            report_text = report_text.strip()
            if not report_text:
                continue 
                
            text_lower = report_text.lower()
            
            if "consolidation" in text_lower or "lobar" in text_lower or "air bronchogram" in text_lower:
                mock_score = "[Normal: 0.0500, Bacterial: 0.8800, Viral: 0.0500, COVID-19: 0.0200]"
            elif "ground-glass" in text_lower or "peripheral" in text_lower or "covid" in text_lower:
                mock_score = "[Normal: 0.0400, Bacterial: 0.1000, Viral: 0.1600, COVID-19: 0.7000]"
            elif "interstitial" in text_lower or "diffuse" in text_lower or "bilateral" in text_lower or "reticular" in text_lower:
                mock_score = "[Normal: 0.1000, Bacterial: 0.1500, Viral: 0.7000, COVID-19: 0.0500]"
            else:
                mock_score = "[Normal: 0.9200, Bacterial: 0.0400, Viral: 0.0300, COVID-19: 0.0100]"

            formatted_data.append({
                "instruction": "You are an expert clinical AI agent. Based on the vision model's classification probabilities, generate a highly professional and concise radiology impression.",
                "input": f"Vision Model Output: {mock_score}",
                "output": f"Impression: {report_text}"
            })
            
        except Exception:
            continue

#data for training
output_file = "med_training_data.json"
with open(output_file, "w") as f:
    json.dump(formatted_data, f, indent=4)

print(f"Success! Generated {len(formatted_data)} high-quality training rows and saved to {output_file}.")