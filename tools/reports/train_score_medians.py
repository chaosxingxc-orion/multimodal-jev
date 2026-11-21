"""Descriptive medians on the complete authenticated clean TRAIN reference."""
import argparse
import json
from statistics import median

from multimodal_jev.local_config import load_config
from tools.reports.jev_train_confusions import run


def summarize(config):
    winners, margins, backgrounds = [], [], []

    def observe(uid, gold, labels, probabilities):
        ordered = sorted(probabilities, reverse=True)
        winners.append(ordered[0])
        margins.append(ordered[0] - ordered[1])
        backgrounds.append(median(ordered))

    audit = run(config, observe=observe)
    return dict(policy=audit['policy_id'], reference=audit['reference_sha256'], recordings=len(winners),
                median_winner=median(winners), median_margin=median(margins),
                median_candidate_background=median(backgrounds), scope='in-sample descriptive TRAIN reference')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('local')
    args = parser.parse_args()
    print(json.dumps(summarize(load_config(args.config, args.local)[0]), indent=2))
