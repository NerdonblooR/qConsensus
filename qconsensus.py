import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import random

PNUM = 10
DOMAIN = 5
SAMPLE_SIZE = 1000
TEST_NUM = 30
SIGMA = 1


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
    next_value  = v+1

    while next != mid:
        if next_value == num:
            next_value = 0
        if next == num:
            next = 0
        option[next] = next_value
        next_value += 1
        next += 1

    return  option


def decide(inputs, prob):
    decisions = []
    #decide based on a normal distribution
    for v in inputs:
        option = generate_option(v, DOMAIN)
        decisions.append(random.choices(population=option, weights=list(prob))[0])
    return decisions

def simulate_different_sigma():
    probs = []
    for sigma in [0.01, 0.5,  5]:
        c = np.random.normal(0, sigma, 1000)
        prob, _, _ = plt.hist(c, DOMAIN, density=True)
        probs.append(prob)
        plt.clf()



    for prob in probs:
        xaxis = []
        yaxis = []
        for j in range(TEST_NUM):
            v_count = 0
            a_count = 0
            for i in range(SAMPLE_SIZE):
                inputs = initialize_input()
                decisions = decide(inputs, prob)
                v_count += is_valid(inputs, decisions)
                a_count += is_agreement(decisions)

            v_rate = v_count / SAMPLE_SIZE
            a_rate = a_count / SAMPLE_SIZE
            xaxis.append(a_rate)
            yaxis.append(v_rate)

            print(v_rate,a_rate)

        plt.scatter(xaxis,yaxis)
        plt.plot(np.unique(xaxis), np.poly1d(np.polyfit(xaxis, yaxis, 1))(np.unique(xaxis)))

    plt.show()



if __name__ == '__main__':
    simulate_different_sigma()


