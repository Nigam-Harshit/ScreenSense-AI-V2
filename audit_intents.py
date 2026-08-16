import json
with open('data/intents_expanded.json') as f:
    data = json.load(f)
print(f"{'Intent':<25} Examples")
print("-" * 38)
for item in data['intents']:
    print(f"{item['intent']:<25} {len(item['examples'])}")
total = sum(len(i['examples']) for i in data['intents'])
print(f"\nTotal: {total} examples across {len(data['intents'])} intents")
print(f"Avg per intent: {total/len(data['intents']):.1f}")
