import math
import csv
import os
import random
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt


random.seed(1)

# Each player uses EW with specified epsilon.


def simulate_auction(n, k, n_players, epsilon_list=[1, 1], value_list=[0.5, 0.5]):
    # Players numbered 0, ..., n_players-1.
    # v[i][j][l] = return of player i on round j if played action l, assuming other players unchanged.
    v = np.zeros((n_players, n+1, k))

    # V[i][j][l] = historical return of player i until round j if he always played action l.
    V = np.zeros_like(v)

    # Auction value and price are in [0,1], divided into 1/k, ..., (k-1)k.
    actions = [[j/k for j in range(k)] for i in range(n_players)]
    n_actions = []  # Number of actions player i can play
    for i in range(n_players):
        n_actions .append(int(value_list[i] * k))
        # DON'T bid higher than your value
        actions[i] = actions[i][0:n_actions[i]]
    moves = np.zeros((n_players, 1))
    payoffs = np.zeros((n_players, n+1))

    # ALL THE WORK
    for round in range(1, n+1):
        does = []
        for i in range(n_players):
            # h is assumed to be 1.
            pitemp = [(1+epsilon_list[i]) ** (V[i][round-1][j])
                      for j in range(n_actions[i])]
            denom = sum(pitemp)
            pitemp = [pitemp[j] / denom for j in range(n_actions[i])]
            do = random.choices(actions[i], weights=pitemp, k=1)
            does.append(do)
        moves = np.append(moves, does, axis=1)

        # Assigning real payoffs
        dealprice = np.amax(moves[:, -1])
        winners = []
        for i in range(n_players):
            if moves[i][-1] == dealprice:
                winners.append(i)
        for i in range(n_players):
            if i in winners:
                # Assign to them their expected payoff
                payoffs[i][round] = payoffs[i][round-1] + \
                    (value_list[i] - dealprice) / len(winners)
            else:
                payoffs[i][round] = payoffs[i][round-1]

        # Assigning potential payoffs
        for i in range(n_players):
            for j in range(n_actions[i]):
                if actions[i][j] < dealprice:
                    v[i][round][j] = 0
                elif actions[i][j] == dealprice:
                    if i in winners:
                        v[i][round][j] = (
                            value_list[i] - actions[i][j]) / len(winners)
                    else:
                        # there is one more winner!
                        v[i][round][j] = (
                            value_list[i] - actions[i][j]) / (len(winners)+1)
                else:  # You are the only winner!
                    v[i][round][j] = value_list[i] - actions[i][j]

        # Calculating emperical payoff V
        for i in range(n_players):
            for j in range(n_actions[i]):
                V[i][round][j] = V[i][round-1][j] + v[i][round][j]
            # Decrease all V for player i uniformly.
            V[i][round] -= np.amin(V[i][round])

    return moves, payoffs, v, V

# multiple players, same value, different epsilon


def test_epsilon(N, n, k, n_players, epsilon_list, value, title):
    V_list = []
    payoffs_list = []
    moves_list = []
    for t in range(N):
        moves, payoffs, v, V = simulate_auction(n, k, n_players, epsilon_list,
                                                value_list=[value for i in range(n_players)])
        V_list.append(V[:, -1])
        payoffs_list.append(payoffs)
        moves_list.append(moves)
    V_list = np.array(V_list)
    V_avg = np.average(V_list, axis=0)
    moves_avg = np.average(moves_list, axis=(0, 2))
    payoffs_list = np.array(payoffs_list)

    # payoff vs round
    plt.figure()
    payoffs_avg = np.average(payoffs_list, axis=0)
    for i in range(n_players):
        plt.plot(payoffs_avg[i], label='lr={}'.format(epsilon_list[i]))
    plt.xlabel("# round")
    plt.ylabel("payoff until current round")
    plt.title("Payoffs vs round, n={}".format(n))
    plt.legend(ncol=n_players, loc='upper center')
    plt.savefig(
        'Online-Market-Project-3/{}, payoff v. round, n={}.png'.format(title, n))

    # payoff vs test run
    plt.figure()
    for i in range(n_players):
        plt.plot(payoffs_list[:, i, -1], label='lr={}'.format(epsilon_list[i]))
    plt.xticks(np.arange(0, 40, step=10))
    plt.xlabel("# test run")
    plt.ylabel("total payoff")
    plt.title("Payoffs each test run, n={}".format(n))
    plt.legend(ncol=n_players, loc='upper center')
    plt.savefig('Online-Market-Project-3/{}, n={}.png'.format(title, n))

    # average bid vs player
    plt.figure()
    plt.bar(np.arange(0, n_players, step=1), moves_avg)
    plt.yticks(np.arange(0, 1, step=0.2))
    plt.xlabel("player #")
    plt.ylabel("average bid")
    plt.title("Average bid over test runs, n={}".format(n))
    plt.savefig('Online-Market-Project-3/{}2, n={}.png'.format(title, n))

    # player 0's choose prabability
    plt.figure()
    plt.plot(np.arange(0, 1, step=0.02), V_avg[0])
    plt.xticks(np.arange(0, 1, step=0.1))
    plt.xlabel("bid")
    plt.ylabel("(relative) total payoff")
    plt.title("total payoff vs bid, n={}".format(n))
    plt.savefig(
        'Online-Market-Project-3/total-payoff-test-epsilon, n={}'.format(n))


def test_value(N, n, k, n_players, epsilon, value_list, title):
    V_list = []
    payoffs_list = []
    moves_list = []
    for t in range(N):
        moves, payoffs, v, V = simulate_auction(n, k, n_players,
                                                epsilon_list=[
                                                    epsilon for i in range(n_players)],
                                                value_list=value_list)
        V_list.append(V[:, -1])
        payoffs_list.append(payoffs)
        moves_list.append(moves)
    V_list = np.array(V_list)
    V_avg = np.average(V_list, axis=0)
    moves_avg = np.average(moves_list, axis=(0, 2))
    payoffs_list = np.array(payoffs_list)

    # payoff vs round
    plt.figure()
    payoffs_avg = np.average(payoffs_list, axis=0)
    for i in range(n_players):
        plt.plot(payoffs_avg[i], label='value={}'.format(value_list[i]))
    plt.xlabel("# round")
    plt.ylabel("payoff until current round")
    plt.title("Payoffs vs round, n={}".format(n))
    plt.legend(ncol=n_players, loc='upper center')
    plt.savefig(
        'Online-Market-Project-3/{}, payoff v. round, n={}.png'.format(title, n))

    # payoff vs test run
    # plt.figure()
    # for i in range(n_players):
    #     plt.plot(payoffs_list[:, i, -1],
    #              label='value={}'.format(value_list[i]))
    # plt.xticks(np.arange(0, 40, step=10))
    # plt.xlabel("# test run")
    # plt.ylabel("total payoff")
    # plt.title("Payoffs each test run, n={}".format(n))
    # plt.legend(ncol=n_players, loc='upper center')
    # plt.savefig('Online-Market-Project-3/{}, n={}.png'.format(title, n))

    # average bid vs player
    # plt.figure()
    # plt.bar(np.arange(0, n_players, step=1), moves_avg)
    # plt.yticks(np.arange(0, 1, step=0.2))
    # plt.xlabel("player #")
    # plt.ylabel("average bid")
    # plt.title("Average bid over test runs, n={}".format(n))
    # plt.savefig('Online-Market-Project-3/{}2, n={}.png'.format(title, n))

    # player 5's choose prabability
    plt.figure()
    plt.plot(np.arange(0, 1, step=0.02), V_avg[-1])
    plt.xticks(np.arange(0, 1, step=0.1))
    plt.xlabel("bid")
    plt.ylabel("(relative) total payoff")
    plt.title("total payoff vs bid, n={}".format(n))
    plt.savefig(
        'Online-Market-Project-3/values-player5, n={}'.format(n))

    # player 0's choose prabability
    plt.figure()
    plt.plot(np.arange(0, 1, step=0.02), V_avg[0])
    plt.xticks(np.arange(0, 1, step=0.1))
    plt.xlabel("bid")
    plt.ylabel("(relative) total payoff")
    plt.title("total payoff vs bid, n={}".format(n))
    plt.savefig(
        'Online-Market-Project-3/values-player0, n={}'.format(n))


# # Test-epsilon
# test_epsilon(N=40, n=5000, k=50, n_players=5,
#              epsilon_list=[1, 2, 3, 4, 5], value=0.5, title='test_epsilon')
# # Test-epsilon-2
# test_epsilon(N=40, n=5000, k=50, n_players=5,
#              epsilon_list=[1, 1, 1, 1, 5], value=0.5, title='test_epsilon+')

#
test_value(N=40, n=5000, k=50, n_players=5, epsilon=1,
           value_list=[0.3, 0.4, 0.5, 0.6, 0.7], title='test_value')


# n=100: 0.25963366 0.25725743 0.25846535 0.25636634 0.32857426]
# n=10000: 0.43436196 0.43438956 0.43429257 0.43440806 0.43880952
