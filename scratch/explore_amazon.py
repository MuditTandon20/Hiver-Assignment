import sys
import urllib.request, csv, io
import pandas as pd

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

url = 'https://huggingface.co/datasets/SunidhiSriram/twcs/resolve/main/twcs.csv'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

lines = []
print("Streaming 120,000 lines from dataset...")
with urllib.request.urlopen(req) as resp:
    for i, line in enumerate(resp):
        lines.append(line.decode('utf-8', errors='ignore'))
        if i >= 120000:
            break

reader = csv.reader(io.StringIO(''.join(lines)))
header = next(reader)
data = [row for row in reader if len(row) == len(header)]
df = pd.DataFrame(data, columns=header)

# Look at tweets involving AmazonHelp
amz_responses = df[df['author_id'] == 'AmazonHelp']
print(f"Total AmazonHelp responses in sample: {len(amz_responses)}")

tweet_dict = {row['tweet_id']: row for _, row in df.iterrows()}

print("\n--- SAMPLE CONVERSATION PAIRS ---")
found = 0
for _, resp_row in amz_responses.iterrows():
    parent_id = resp_row['in_response_to_tweet_id']
    if parent_id and parent_id in tweet_dict:
        cust_row = tweet_dict[parent_id]
        if cust_row['inbound'] == 'True':
            found += 1
            print(f"\n[PAIR #{found}] Tweet ID: {cust_row['tweet_id']} -> Reply ID: {resp_row['tweet_id']}")
            print(f"  CUSTOMER: {cust_row['text']}")
            print(f"  AMAZON:   {resp_row['text']}")
            if found >= 15:
                break
