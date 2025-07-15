## Creating seed instructions

### `seed_instructions_ifeval.txt`

Based on 25 IFEval seed verifiable instructions (Table1 in https://arxiv.org/pdf/2311.07911). The data is manually parsed from Table1 and ammended with 5 handcrafted examples per instruction saved in `ifeval_verifiable_instructions.json`. 

To convert `ifeval_verifiable_instructions.json` into the current `seed_instructions_ifeval.txt` run

```sh
python3 -c "import json, random; data = json.load(open('data/ifeval_verifiable_instructions.json')); open('data/seed_instructions_ifeval.txt', 'w').write('\n'.join([random.choice(item['examples']) for item in data]))"
```