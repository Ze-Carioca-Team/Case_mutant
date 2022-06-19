import json
import math
import random
import argparse
from eda import eda
from tqdm import tqdm
from collections import defaultdict
from genentities import gen_pizza, gen_text_from_pizza, slot_value_endereco

random.seed(20211109)

def augment(sentence):
    return eda(sentence, ["[itens]", "[endereco]"])

def replicate(flows, utterances, rate):
    samples = []
    for flow in flows:
        for _ in range(rate):
            curr = []
            for i, f in enumerate(flow):
                curr.append(random.choice(utterances["action" if not i%2 else "intent"][f]))
            samples.append(curr)
    return samples


def parse_args():
    parser = argparse.ArgumentParser(description="Applying MADA on a dialog dataset formatted in the MultiWOZ pattern.")
    parser.add_argument("--filename", type=str, default="dialogs.json", help="Path to dialogs dataset.")
    parser.add_argument("--rate", type=int, default=100, help="Replication.")
    parser.add_argument("--sample-size", dest='sample', type=int, default=5000, help="Size of sample to pick.")
    parser.add_argument("--no-augment", default=True, help="Augment dataset.", dest='augment', action="store_false")
    return parser.parse_args()

def main():
    args = parse_args()
    possible_flows = {}
    utt_counter = defaultdict(int)
    intent_sample = {"intent":defaultdict(list), "action":defaultdict(list)}
    with open(args.filename) as fin:
        data = json.load(fin)
    random.shuffle(data["dialogs"])
    for dialog in data["dialogs"]:
        dialog["id"] = str(dialog["id"])
        curr_flow = []
        for i, turn in enumerate(dialog["turns"]):
            agent = "intent" if i % 2 == 1 else "action"
            utt = turn[agent]
            utt_counter[utt] += 1
            curr_flow.append(utt)
            intent_sample[agent][utt].append(turn)
        curr_flow = tuple(curr_flow)
        if curr_flow in possible_flows:
            possible_flows[curr_flow] += 1
        else:
            possible_flows[curr_flow] = 1
    counterf = defaultdict(int)
    for k,v in possible_flows.items():
        counterf[k[0]] += v
    print(counterf)
    for count in sorted(utt_counter.items(), key=lambda x: x[1]):
        print(count)
    samples = replicate(possible_flows.keys(), intent_sample, args.rate)
    out_data = []
    for i, dialog in enumerate(tqdm(samples)):
        new_dialog = []
        current_values = {}
        for num, turn in enumerate(dialog):
            new_turn = turn.copy()
            new_turn["turn-num"] = num
            if turn["speaker"] == "client":
                if args.augment and random.random() >= .5:
                    random.seed(20211109+i)
                    aug_text = augment(new_turn["utterance_delex"].lower())
                    new_turn["utterance_delex"] = random.choice(aug_text)
                else:
                    new_turn["utterance_delex"] = new_turn["utterance_delex"].lower()
            new_turn["utterance"] = new_turn["utterance_delex"]
            if "[itens]" in new_turn["utterance"]:
                pizza = gen_pizza()
                current_values["itens"] = " ".join([" ".join(x) for x in pizza])
                pizzatxt = gen_text_from_pizza(pizza)
                new_turn["utterance"] = new_turn["utterance"].replace("[itens]", pizzatxt)
            if "[endereco]" in new_turn["utterance"]:
                end = slot_value_endereco()
                current_values["endereco"] = end
                new_turn["utterance"] = new_turn["utterance"].replace("[endereco]", end)
            if turn["speaker"] == "client":
                new_turn["slot-values"] = current_values.copy()
            new_dialog.append(new_turn)
        out_data.append({
            "id": f"{i*1000}",
            "turns": new_dialog})
    random.shuffle(out_data)
    data["dialogs"] = out_data
    with open("out."+args.filename, "w") as fout:
        json.dump(data, fout, indent=2, ensure_ascii=False, sort_keys=True)


if __name__ == "__main__":
    main()
