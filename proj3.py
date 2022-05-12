import math
import csv
import os
import random
import numpy as np
import matplotlib.pyplot as plt


random.seed(1)

# Each player uses EW with specified epsilon.


def EW(k, epsilon, actions, V):
    pitemp = [(1+epsilon) ** (V[j])
              for j in range(len(actions))]
    denom = sum(pitemp)
    pitemp = [pitemp[j] / denom for j in range(len(actions))]
    do = random.choices(actions, weights=pitemp, k=1)
    return do, pitemp

# EPSILON < 1


def MAB(k, epsilon, actions, V):
    pitemp = [(1+epsilon) ** (V[j])
              for j in range(len(actions))]
    denom = sum(pitemp)
    pitemp = [pitemp[j] / denom for j in range(len(actions))]
    pitemp = [pitemp[j] * (1-epsilon) + epsilon /
              k for j in range(len(actions))]
    do = random.choices(actions, weights=pitemp, k=1)
    return do, pitemp


def simulate_auction(n, k, n_players, epsilon_list, value_list):
    # Players numbered 0, ..., n_players-1.
    # v[i][j][l] = return of player i on round j if played action l, assuming other players unchanged.
    v = np.zeros((n_players, n+1, k))

    # V[i][j][l] = historical return of player i until round j if he always played action l.
    V = np.zeros_like(v)

    converged = [False for i in range(n_players)]

    # Auction value and price are in [0,1], divided into 0, 1/k, ..., (k-1)k.
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
            Vtemp = V[i][round-1] - np.amin(V[i][round-1])
            # h is assumed to be 1.
            pitemp = [(1+epsilon_list[i]) ** (V[i][round-1][j])
                      for j in range(len(actions[i]))]
            denom = sum(pitemp)
            pitemp = [pitemp[j] / denom for j in range(len(actions[i]))]

            if np.amax(pitemp) > 0.98 and converged[i] is False:
                print('player {} converged at round {}'.format(i, round))
                converged[i] = True
            do = random.choices(actions[i], weights=pitemp, k=1)
            does.append(do)
        moves = np.append(moves, does, axis=1)

        # Calculatng winner
        dealprice = np.amax(moves[:, -1])
        winners = []
        for i in range(n_players):
            if moves[i][-1] == dealprice:
                winners.append(i)

        for i in range(n_players):
            payoffs[i][round] = payoffs[i][round-1]

        # Assigning real payoffs
        # for winner in winners:
        #     # Assign to them their expected payoff
        #     payoffs[i][round] = payoffs[i][round-1] + \
        #         (value_list[i] - dealprice) / len(winners)

        # AlTERNATIVE REAL PAYOFFS
        winner = winners[random.randint(0, len(winners)-1)]
        payoffs[winner][round] += value_list[winner] - dealprice

        # Assigning potential payoffs
        for i in range(n_players):
            temp = moves[i][-1]
            for j in range(n_actions[i]):
                # Recalculate the winners in this case
                moves[i][-1] = actions[i][j]
                dealprice = np.amax(moves[:, -1])
                winners = []
                for l in range(n_players):
                    if moves[l][-1] == dealprice:
                        winners.append(l)

                # calculate EXPECTED payoff
                if actions[i][j] < dealprice:
                    v[i][round][j] = 0
                elif actions[i][j] == dealprice:
                    v[i][round][j] = (
                        value_list[i] - actions[i][j]) / len(winners)
            moves[i][-1] = temp

        # Calculating emperical payoff V
        for i in range(n_players):
            for j in range(n_actions[i]):
                V[i][round][j] = V[i][round-1][j] + v[i][round][j]
            # Decrease all V for player i uniformly.

    return moves, payoffs, v, V


def manipulate_auction(n, k, n_players, epsilon=[1, 1], value_list=[0.5, 0.5], rounds_to_fool=10, action=0.1):
    # value_list = [opponent's value, our value]

    # v and V are for opponent
    v = np.zeros((n_players-1, n+1, k))
    V = np.zeros_like(v)
    payoffs = np.zeros((n_players, n+1))

    # Auction value and price are in [0,1], divided into 0, 1/k, ..., (k-1)k.
    actions = [[j/k for j in range(k)] for i in range(n_players)]
    n_actions = []  # Number of actions player i can play
    for i in range(n_players):
        n_actions.append(int(value_list[i] * k))
        # DON'T bid higher than your value
        actions[i] = actions[i][0:n_actions[i]]
    # REVERSED ORDER OF INDICES
    moves = moves = [[0 for i in range(n_players)]]
    payoffs = np.zeros((n_players, n+1))

    def foolthem(round):
        if round % rounds_to_fool == 0:
            return action
        else:
            return 0.0

    # ALL THE WORK
    for round in range(1, n+1):
        does = []
        # h is assumed to be 1.
        for i in range(n_players-1):
            Vtemp = V[i][round-1] - np.amin(V[i][round-1])
            pitemp = [(1+epsilon) ** (Vtemp[j])
                      for j in range(n_actions[i])]
            denom = sum(pitemp)
            pitemp = [pitemp[j] / denom for j in range(n_actions[i])]
            do = random.choices(actions[i], weights=pitemp, k=1)
            does.append(do[0])
        does.append(foolthem(round))
        moves.append(does)

        # Calculatng winner
        dealprice = np.amax(moves[-1])
        winners = []
        for i in range(n_players):
            if moves[-1][i] == dealprice:
                winners.append(i)

        for i in range(n_players):
            payoffs[i][round] = payoffs[i][round-1]

        # Assigning real payoffs
        # for winner in winners:
        #     # Assign to them their expected payoff
        #     payoffs[i][round] = payoffs[i][round-1] + \
        #         (value_list[i] - dealprice) / len(winners)

        # AlTERNATIVE REAL PAYOFFS
        winner = winners[random.randint(0, len(winners)-1)]
        payoffs[winner][round] += value_list[winner] - dealprice

        # Assigning potential payoffs
        for i in range(n_players-1):
            temp = moves[-1][i]
            for j in range(n_actions[i]):
                # Recalculate the winners in this case
                moves[-1][i] = actions[i][j]
                dealprice = np.amax(moves[-1])
                winners = []
                for l in range(n_players):
                    if moves[-1][l] == dealprice:
                        winners.append(l)

                # calculate EXPECTED payoff
                if actions[i][j] < dealprice:
                    v[i][round][j] = 0
                elif actions[i][j] == dealprice:
                    v[i][round][j] = (
                        value_list[i] - actions[i][j]) / len(winners)
            moves[-1][i] = temp

        # Calculating emperical payoff V
        for i in range(n_players-1):
            for j in range(n_actions[i]):
                V[i][round][j] = V[i][round-1][j] + v[i][round][j]
            # Decrease all V for player i uniformly.

    return moves, payoffs, v, V

# multiple players, same value, different epsilon


def test_epsilon(N, n, k, n_players, epsilon_list, value, title):
    V_list = []
    payoffs_list = []
    moves_list = []
    for t in range(N):
        moves, payoffs, v, V = simulate_auction(n, k, n_players, epsilon_list,
                                                value_list=[
                                                    value for i in range(n_players)])
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
        'epsilon/{}, payoff v. round, n={}.png'.format(title, n))

    # payoff vs test run
    plt.figure()
    for i in range(n_players):
        plt.plot(payoffs_list[:, i, -1], label='lr={}'.format(epsilon_list[i]))
    plt.xticks(np.arange(0, 40, step=10))
    plt.xlabel("# test run")
    plt.ylabel("total payoff")
    plt.title("Payoffs each test run, n={}".format(n))
    plt.legend(ncol=n_players, loc='upper center')
    plt.savefig('epsilon/{}, n={}.png'.format(title, n))

    # average bid vs player
    # plt.figure()
    # plt.bar(np.arange(0, n_players, step=1), moves_avg)
    # plt.yticks(np.arange(0, 1, step=0.2))
    # plt.xlabel("player #")
    # plt.ylabel("average bid")
    # plt.title("Average bid over test runs, n={}".format(n))
    # plt.savefig('epsilon/{}2, n={}.png'.format(title, n))

    # players' choose prabability
    plt.figure()
    for i in range(n_players):
        Vtemp = V_avg[i] - np.amin(V_avg[i])
        pitemp = [(1+epsilon_list[i]) ** (Vtemp[j])
                  for j in range(int(value*k))]
        denom = sum(pitemp)
        prob = [pitemp[j] / denom for j in range(int(value*k))]
        while len(prob) < k:
            prob.append(0)
        plt.plot(np.arange(0, 1, step=0.02), prob, label='player {}'.format(i))
    plt.xticks(np.arange(0, 1, step=0.1))
    plt.xlabel("bid")
    plt.ylabel("probabiltiy")
    plt.title("probability vs bid, n={}".format(n))
    plt.legend()
    plt.savefig(
        'epsilon/choose-prob-{}, n={}'.format(title, n))


def test_value(N, n, k, n_players, epsilon, value_list, title):
    V_list = []
    payoffs_list = []
    moves_list = []
    for t in range(N):
        moves, payoffs, v, V = simulate_auction(n, k, n_players,
                                                epsilon_list=[
                                                    epsilon for i in range(n_players)],
                                                value_list=value_list)
        V_list.append(V[:, -1, :])
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
    plt.figure()
    for i in range(n_players):
        plt.plot(payoffs_list[:, i, -1],
                 label='value={}'.format(value_list[i]))
    plt.xticks(np.arange(0, 40, step=10))
    plt.xlabel("# test run")
    plt.ylabel("total payoff")
    plt.title("Payoffs each test run, n={}".format(n))
    plt.legend(ncol=n_players, loc='upper center')
    plt.savefig('Online-Market-Project-3/{}, n={}.png'.format(title, n))

    # players' choose prabability
    plt.figure()
    for i in range(n_players):
        Vtemp = V_avg[i] - np.amin(V_avg[i])
        pitemp = [(1+epsilon) ** (Vtemp[j])
                  for j in range(int(value_list[i]*k))]
        denom = sum(pitemp)
        prob = [pitemp[j] / denom for j in range(int(value_list[i]*k))]
        while len(prob) < k:
            prob.append(0)
        plt.plot(np.arange(0, 1, step=0.02), prob, label='player {}'.format(i))
    plt.xticks(np.arange(0, 1, step=0.1))
    plt.xlabel("bid")
    plt.ylabel("probabiltiy")
    plt.title("probability vs bid, n={}".format(n))
    plt.legend()
    plt.savefig(
        'Online-Market-Project-3/choose-prob-{}, n={}'.format(n, title))


def test_adversarial(n, k, n_players, epsilon, value_list, title):
    moves, payoffs, v, V = manipulate_auction(
        n, k, n_players, epsilon=epsilon, value_list=value_list)
    # payoff vs round
    plt.figure()
    for i in range(n_players-1):
        plt.plot(payoffs[i], label='opponent {}'.format(i))
    plt.plot(payoffs[-1], label='us')
    plt.xlabel("# round")
    plt.ylabel("payoff until current round")
    plt.title("Payoffs vs round, n={}".format(n))
    plt.legend(ncol=2, loc='upper center')
    plt.savefig(
        'adversarial/{}, payoff v. round, n={}.png'.format(title, n))

    plt.figure()
    plt.plot(payoffs[-1], label='us')
    plt.xlabel("# round")
    plt.ylabel("payoff until current round")
    plt.title("Payoffs vs round, n={}".format(n))
    plt.legend(ncol=2, loc='upper center')
    plt.savefig(
        'adversarial/us, payoff v. round, n={}.png'.format(title, n))

    # player 0's choose prabability
    plt.figure()
    plt.plot(np.arange(0, 1, step=0.02), V[0][-1])
    plt.xticks(np.arange(0, 1, step=0.1))
    plt.xlabel("bid")
    plt.ylabel("(relative) total payoff")
    plt.title("total payoff vs bid, n={}".format(n))
    plt.savefig(
        'adversarial/{}, opponent-choices, n={}'.format(title, n))


def test_adversarial_plus(n, k, n_players, epsilon, value_list):
    actions = np.arange(0, value_list[-1], step=1/k)
    moves = np.zeros((len(actions), 20))
    payoffs = np.zeros_like(moves)
    for i, a in enumerate(actions):
        for j in range(1, 20):
            move, payoff, v, V = manipulate_auction(n, k, n_players, epsilon=epsilon, value_list=value_list,
                                                    rounds_to_fool=j, action=a)
            payoffs[i][j] = payoff[1][-1]

    # Heatmap
    plt.figure()
    plt.imshow(payoffs)
    plt.xlabel('lambda')
    plt.ylabel('A')
    plt.xticks([0, 5, 10, 15, 20], [0, 5, 10, 15, 20])
    plt.yticks([0, 2, 4, 6, 8, 10, 12, 14], [
               0, 0.04, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28])
    plt.colorbar()
    plt.title('total payoff after 200 rounds for different parameters')
    plt.savefig('adversarial/heatmap, n={}'.format(n))


# Test-epsilon
# test_epsilon(N=40, n=5000, k=50, n_players=5,
#              epsilon_list=[1, 2, 3, 4, 5], value=0.5, title='test_epsilon')
# # Test-epsilon-2
# test_epsilon(N=40, n=1000, k=50, n_players=5,
#              epsilon_list=[1, 1, 1, 1, 5], value=0.5, title='test_epsilon+')
plt.figure()
plt.plot([1, 2, 3, 4, 5], [2029, 1066, 824, 692, 641])
plt.xticks(np.arange(1, 6, step=1))
plt.xlabel('epsilon')
plt.ylabel('# rounds')
plt.title('number of rounds before convergence (p>0.98)')
plt.savefig('convergence')

# test_value(N=40, n=5000, k=50, n_players=5, epsilon=1,
#            value_list=[0.1, 0.2, 0.3, 0.4, 0.9], title='test_value')


# test_adversarial(n=200, k=50, n_players=2, epsilon=1, value_list=[
#                  0.6, 0.3], title='adversarial')
# test_adversarial(n=200, k=50, n_players=5, epsilon=1, value_list=[
#                  0.6, 0.6, 0.6, 0.6, 0.3], title='adversarial+')
test_adversarial_plus(n=200, k=50, n_players=2,
                      epsilon=1, value_list=[0.6, 0.3])
