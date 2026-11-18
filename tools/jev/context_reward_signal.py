"""Exact reward distribution for independent Bernoulli, singleton event gold.

Uses a Poisson-binomial recurrence, without Monte Carlo or model execution.
Only applicable to complete exclusive-event truth and factorized sampling.
"""
import math


def singleton_signal(probabilities,gold_index,*,group_size=16):
    if (not probabilities or type(gold_index) is not int or not 0<=gold_index<len(probabilities) or
            type(group_size) is not int or group_size<2 or
            any(type(p) not in (int,float) or not math.isfinite(p) or not 0<=p<=1 for p in probabilities)):
        raise ValueError('finite Bernoulli probabilities, one gold index and group size required')
    count=[1.]
    for index,p in enumerate(probabilities):
        if index==gold_index:continue
        next_count=[0.]*(len(count)+1)
        for k,mass in enumerate(count):
            next_count[k]+=mass*(1-p);next_count[k+1]+=mass*p
        count=next_count
    positive=probabilities[gold_index]
    # If gold is absent F1=0. With gold present and k false positives, F1=2/(k+2).
    masses=[(0.,1-positive)]+[(2/(k+2),positive*mass) for k,mass in enumerate(count)]
    mean=sum(reward*mass for reward,mass in masses)
    variance=sum(mass*(reward-mean)**2 for reward,mass in masses)
    zero_signal=sum(mass**group_size for _,mass in masses)
    empty=math.prod(1-p for p in probabilities)
    return dict(expected_set_f1=mean,reward_variance=variance,
        group_all_equal_reward_probability=zero_signal,
        group_positive_reward_probability=1-(1-positive)**group_size,
        exact_correct_set_probability=positive*count[0],empty_set_probability=empty,
        multiple_selection_probability=max(0.,min(1.,1-empty-positive*count[0]-(1-positive)*(count[1] if len(count)>1 else 0.))),
        expected_selected_count=sum(probabilities),gold_probability=positive,
        group_size=group_size,candidates=len(probabilities))
