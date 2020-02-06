import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import random

PNUM = 10
DOMAIN = 4
SAMPLE_SIZE = 1000
TEST_NUM = 50


def initialize_input():
    inputs = []
    domain_list = list(range(DOMAIN))
    for i in range(PNUM):
        inputs.append(random.choice(domain_list))
    return inputs


def is_valid(inputs, decisions):
    iset = set(inputs)
    dset = set(decisions)
    return dset.issubset(iset)


def is_agreement(decisions):
    return len(set(decisions)) == 1


def generate_option(v, num):
    option = [0] * num
    mid = num // 2
    next = mid + 1
    option[mid] = v
    next_value = v + 1
    while next != mid:
        if next_value == num:
            next_value = 0
        if next == num:
            next = 0
        option[next] = next_value
        next_value += 1
        next += 1

    return option


def decide(inputs, prob, mode):
    decisions = []
    for v in inputs:
        if mode:
            option = generate_option(v, DOMAIN)
        else:
            option = list(range(DOMAIN))
        decisions.append(random.choices(population=option, weights=list(prob))[0])
    return decisions


def simulate_consensus():
    probs = []
    labels = ['normal', 'uniform', 'one']
    # normal distribution
    ssize = 1000
    c = np.random.normal(0, 0.5, ssize)
    prob, _, _ = plt.hist(c, DOMAIN)
    prob = list(map(lambda x: x / ssize, prob))
    probs.append(prob)
    plt.clf()

    # uniform distribution
    prob = [1 / DOMAIN] * DOMAIN
    probs.append(prob)

    # uniform distribution
    prob = [0] * DOMAIN
    prob[DOMAIN // 2] = 1
    probs.append(prob)

    for i in range(len(probs)):
        prob = probs[i]
        print(prob)
        xaxis = []
        yaxis = []
        for j in range(TEST_NUM):
            v_count = 0
            a_count = 0
            for k in range(SAMPLE_SIZE):
                inputs = initialize_input()
                decisions = decide(inputs, prob, 0)
                v_count += is_valid(inputs, decisions)
                a_count += is_agreement(decisions)
            v_rate = v_count / SAMPLE_SIZE
            a_rate = a_count / SAMPLE_SIZE
            xaxis.append(a_rate)
            yaxis.append(v_rate)

            print(v_rate, a_rate)

        plt.scatter(xaxis, yaxis, label=labels[i])
        plt.legend()

    labels = ['normal_ir', 'uniform_ir', 'one_ir']
    for i in range(len(probs)):
        prob = probs[i]
        print(prob)
        xaxis = []
        yaxis = []
        for j in range(TEST_NUM):
            v_count = 0
            a_count = 0
            for k in range(SAMPLE_SIZE):
                inputs = initialize_input()
                decisions = decide(inputs, prob, 1)
                v_count += is_valid(inputs, decisions)
                a_count += is_agreement(decisions)
            v_rate = v_count / SAMPLE_SIZE
            a_rate = a_count / SAMPLE_SIZE
            xaxis.append(a_rate)
            yaxis.append(v_rate)

            print(v_rate, a_rate)

        plt.scatter(xaxis, yaxis, label=labels[i])
        plt.legend(loc='upper center')

if __name__ == '__main__':
    simulate_consensus()
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.xlabel('P(A)')
    plt.ylabel('P(V)')
    plt.title("PROCESS_NUM={0}, DOMAIN_SIZE={1}".format(PNUM, DOMAIN))
    plt.show()
